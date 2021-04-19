from __future__ import division
from __future__ import print_function
from graphfeatures import graph_norm
import numpy as np
import tensorflow as tf
import time
from copy import deepcopy
from sklearn import svm
from Graph_layers import GraphConvolution, GraphLinear, InnerProductDecoder
from clustering import A_binarize, creating_label
from pre_func import dataset2_indices, preprocess_data
from evaluation import EER_calculation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

loss_function = 3
decoder_adj = True
def invlogit(z):
    return 1 - 1 /(1 + np.exp(z))
    
class GCNModelVAE(tf.keras.Model):
    def __init__(self, num_features, num_nodes, features_nonzero, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.hidden_dim = 32 #32
        self.hidden_dim2 = 32
        if(loss_function==1 or loss_function==3):
            self.dimension = 1
        else:
            self.dimension = 4
        self.hidden1 = GraphConvolution(input_dim = self.input_dim, 
                                        output_dim = self.hidden_dim, num = 1,
                                        act = lambda x: x)
        """
        self.hidden12 = GraphConvolution(input_dim = self.hidden_dim, 
                                        output_dim = self.hidden_dim, num = 4,
                                        act = tf.nn.relu)
        #"""
        self.hidden2 = GraphConvolution(input_dim = self.hidden_dim,
                                        output_dim = self.dimension*2, num = 2,
                                        act = lambda x: x)
        self.d = InnerProductDecoder(act = lambda x: x)
        if(loss_function==1 or loss_function==3):
            if(decoder_adj):
                self.d1 = GraphConvolution(input_dim = 1,
                                       output_dim = self.n_samples, num = 3,
                                       act = lambda x: x)
            else:
                self.d1 = GraphConvolution(input_dim = self.n_samples,
                                       output_dim = self.n_samples, num = 3,
                                       act = lambda x: x)
    def encoder(self, inputs, adj, rate):
        x = self.hidden1(inputs, adj, rate)
        #x = tf.keras.layers.BatchNormalization()(x)
        #x = self.hidden12(x, adj, rate)
        x = self.hidden2(x, adj, rate)
        mean, logvar = tf.split(x, num_or_size_splits=2, axis=2)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal([self.n_samples, self.dimension])
        return eps * (tf.exp(logvar)) + mean
    
    def decoder(self, z, adj, rate=0., apply_sigmoid=False):
        logits = z
        logits = self.d(logits,0.)
        if(loss_function==1 or loss_function==3):
            if(decoder_adj):
                feature = tf.ones((logits.shape[0],logits.shape[1],1))
                logits = self.d1(feature,logits,rate)
            else:
                logits = self.d1(logits,adj,rate)
        logits = tf.reshape(logits, [-1,self.n_samples*self.n_samples])
        if apply_sigmoid:
          probs = tf.sigmoid(logits)
          return probs
        return logits

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=1e-4,
                decay_steps=10000,
                decay_rate=0.9) #opt.optimizer._decayed_lr(tf.float32)

class OptimizerVAE(object):
    def __init__(self, model, num_nodes,num_features,norm):
        self.norm = norm
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)
    def log_normal_pdf(self,sample, mean, logsd, raxis=[1,2]):
        logvar = 2 * logsd
        log2pi = tf.math.log(2. * np.pi)
        out = tf.reduce_sum(-.5 * (tf.multiply((sample - mean) ** 2., tf.exp(-logvar)) + logvar + log2pi),axis=raxis)
        return out    
    def bernoulli_log_density(self,logit,x):
        b = (x * 2) - 1
        return - tf.math.log(1 + tf.exp(-tf.multiply(b,logit)))
    def loss(self,y,x,adj,rate, model):
        mean, logvar = model.encoder(x,adj,rate)
        reparam = model.reparameterize(mean,logvar)
        reconstruct = model.decoder(reparam, adj, rate)
        preds_sub = tf.reshape(reconstruct, [-1,self.num_nodes,self.num_nodes])
        logpz = self.log_normal_pdf(reparam, 0., 0.)
        logqz_x = self.log_normal_pdf(reparam, mean, logvar)
        if(loss_function==3):
            logpx_z = tf.reduce_sum(self.bernoulli_log_density(preds_sub,tf.cast(y,tf.float32)),[1,2])
            return -tf.reduce_mean(logpx_z - ((logpz - logqz_x)))
        else:
            cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(y,tf.float32), logits=tf.cast(preds_sub,tf.float32))
            logpx_z = tf.reduce_sum(cross_ent, axis=[1, 2])
            return tf.reduce_mean(logpx_z + ((logpz - logqz_x)))
    def loss2(self,y, x,adj,rate, model):
        mean, logvar = model.encoder(x,adj,rate)
        reparam = model.reparameterize(mean,logvar)
        reconstruct = model.decoder(reparam, adj, rate)
        preds_sub = tf.reshape(reconstruct, [-1,self.num_nodes,self.num_nodes])
        cost = self.norm * tf.reduce_mean(tf.reduce_sum(
                        tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.cast(y,tf.float32), 
                                                    logits = preds_sub),[1,2]))
        kl = (0.5 / num_nodes) * \
                  tf.reduce_mean(tf.reduce_sum(1 \
                                               + 2 * logvar \
                                               - tf.square(mean) \
                                               - tf.square(tf.exp(logvar)), [1,2]))         
        cost -= kl
        return cost
    def train_step(self,y,x,adj,rate,model):
        with tf.GradientTape() as tape:
            if(loss_function== 3 or loss_function==1):
                cost = self.loss(y,x,adj,rate, model)
            else:
                cost = self.loss2(y,x,adj,rate, model)
        assert not np.any(np.isnan(cost.numpy()))
        gradients = tape.gradient(cost, model.trainable_variables)
        opt_op = self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return cost

channel7 = False
if(channel7):
    Binary = False
    #subject_num = subject_num2
else:
    Binary = True
Part_channel = False; partial_subject = False; part_channel = False

def Adj_matrix(train_x, test_x):   
    if(Binary):
        percentile = 0.75 #.75 datasetII #0.9
        adj_train = A_binarize(A_matrix=train_x,percent=percentile,sparse=False)
        adj_test  = A_binarize(A_matrix=test_x,percent=percentile,sparse=False)
        #sparse matrix
    else:
        adj_train = deepcopy(train_x) 
        adj_test = deepcopy(test_x) 
    if(Part_channel):
        index = creating_label(ztr,y_train,subject_num,method='mean_sort') #dataset2_indices(signal_channel)
        adj_train = adj_train[:,:,index]
        adj_train = adj_train[:,index]
        adj_test = adj_test[:,:,index]
        adj_test = adj_test[:,index]
    #scipy.sparse.issparse(adj_train[9]) #check sparsity
    #np.linalg.matrix_rank(adj_train[9]) #check matrix rank
    return adj_train, adj_test
    
FLAGS_features = False
if not FLAGS_features:
    features_init_train = None
else:
    features_init_train = deepcopy(train_x)
if not FLAGS_features:
    features_init_test = None
else:
    features_init_test = deepcopy(test_x)

verbose = True
nb_run = 1
accuracy = np.zeros((nb_run,1))
accuracy2 = np.zeros((nb_run,1))
Computational_time = np.zeros((nb_run,1))
num_epoch = np.zeros((nb_run,1))
full_time = np.zeros((nb_run,1))
roc_auc = np.zeros((nb_run,1))
EER = np.zeros((nb_run,1))

for i in range(nb_run):
    t_start = time.time()
    if verbose:
        print("Creating Adjacency matrix...")
    """
    train_x, test_x, y_train, y_test = preprocess_data(x2[:,1],Labels,i,Fs,dataset2=False,
                                                      filt=False,ICA=True,A_Matrix='cov',sec=1)    
    #"""
    """
    train_x, test_x, y_train, y_test = preprocess_data(physio_data[:,:n,:],Labels,i,
                                                       Fs,dataset2=False,filt=False,
                                                       ICA=True,A_Matrix='plv',sec=1)
    #"""
    """
    if(channel7):
        train_x, test_x, y_train, y_test = preprocess_data(x_original[:,:,Fs*9:],Labels,i,Fs,
                                                           dataset2=False,filt=True,ICA=True,A_Matrix='cov',sec=1)
    else:
        train_x, test_x, y_train, y_test = preprocess_data(x_original_all[:,0],Labels,i,Fs,dataset2=False,
                                                       filt=False,ICA=True,A_Matrix='cov',sec=12)    
    #train_x, test_x, y_train, y_test = deepcopy(train_x_task), deepcopy(test_x_task), deepcopy(y_train_task), deepcopy(y_test_task)
    #"""
    #"""
    #A_matrix = 'cov' 'plv' 'iplv' 'pli' 'AEC'
    adj_train, adj_test = Adj_matrix(train_x, test_x)
    # Preprocessing and initialization
    if verbose:
        print("Preprocessing and Initializing...")
    # Compute number of nodes
    num_nodes = adj_train.shape[1]
    # If features are not used, replace feature matrix by identity matrix
    I = (np.tile(np.eye(adj_train.shape[1]),adj_train.shape[0]).T).reshape(-1,adj_train.shape[1],adj_train.shape[1])
    I_test = (np.tile(np.eye(adj_test.shape[1]),adj_test.shape[0]).T).reshape(-1,adj_test.shape[1],adj_test.shape[1])
    if not FLAGS_features:
        features = np.ones((adj_train.shape[0],adj_train.shape[1],1))
        #features = deepcopy(I)
    else:
        features = deepcopy(features_init_train)
    # Preprocessing on node features
    num_features = features.shape[2]
    features_nonzero = np.count_nonzero(features)//features.shape[0]
    # Normalization and preprocessing on adjacency matrix
    adj_norm = graph_norm(adj_train)
    adj_label = adj_train + I
    
    adj_norm_test = graph_norm(adj_test)
    adj_label_test = adj_test + I_test
    if not FLAGS_features:
        features_test = np.ones((adj_test.shape[0],adj_test.shape[1],1))
        #features_test = deepcopy(I_test)
    else:
        features_test = deepcopy(features_init_test)
    #"""
    num_part = 19
    if(partial_subject):
        del_index = np.where(y_train>=num_part)[0]
        adj_train_par = np.delete(adj_train,del_index,axis=0)
        adj_n = np.delete(adj_norm,del_index,axis=0)
        adj_l = np.delete(adj_label,del_index,axis=0)
        feat = np.delete(features,del_index,axis=0)
        train_dataset = (tf.data.Dataset.from_tensor_slices((adj_n,adj_l,feat))
                         .shuffle(len(adj_n)).batch(64))
        norm = adj_train_par.shape[1] * adj_train_par.shape[1] / float((adj_train_par.shape[1] * adj_train_par.shape[1]
                                                - (adj_train_par.sum()/adj_train_par.shape[0])) * 2)
    else:
        train_dataset = (tf.data.Dataset.from_tensor_slices((adj_norm,adj_label,features))
                         .shuffle(len(adj_norm)).batch(64))
        norm = adj_train.shape[1] * adj_train.shape[1] / float((adj_train.shape[1] * adj_train.shape[1]
                                                - (adj_train.sum()/adj_train.shape[0])) * 2)
    rate_test = 0
    VAEmodel = GCNModelVAE(num_features, num_nodes,features_nonzero)
    # Optimizer
    opt = OptimizerVAE(model = VAEmodel, num_nodes = num_nodes, 
                       num_features=num_features, norm=norm)
    # Model training
    if verbose:
        print("Training...")
    prev_cost = 100000
    stop_val = 0
    stop_num = 10
    FLAGS_shuffle = False
    if(loss_function == 1):
        if(Binary):
            if(Part_channel):
                n_epochs = 15 #27
            else:
                #29 
                n_epochs = 171#240 for 10 subjects #71 for 30 subjects #22 for all subjects #48 for 50 subject
        else:
            n_epochs = 25 #38 #16
    elif(loss_function==3):
        n_epochs = 16 #14 #43 without d1 
    else:
        n_epochs = 1000
    for epoch in range(2000): #20 #400 & 500 for 8 seconds #134 for 4 second
        t = time.time()
        # Compute average loss
        loss = 0
        for adj, label, x in train_dataset:
            loss += opt.train_step(label,tf.cast(x,tf.float32),tf.cast(adj,tf.float32), 0.5, VAEmodel)
        #loss = opt.train_step(adj_label,tf.cast(features,tf.float32),tf.cast(adj_norm,tf.float32), 0.5, model)
        avg_cost = loss.numpy() / (len(adj_train))
        if verbose:
            # Display epoch information
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(round(avg_cost,3)),
                  "time=", "{:.5f}".format(round(time.time() - t,3)))
        Computational_time[i] += (time.time() - t)
        num_epoch[i] +=1
        """
        if(i==2 or i==3):
            if(avg_cost<=-35): 
                break
        else:
            if(avg_cost<=-9): 
                break
        #"""
        #"""
        if(avg_cost<=-10): #-15 PLV filter #-5 #-90 #-30 #9
                break
        #"""
        """
        if(i==3):
            if(avg_cost<=-10): 
                break
        elif(i==1):
            if(avg_cost<=-4):
                break
        elif(i==2):
            if(avg_cost<=-6): 
                break
        else:
            if(avg_cost<=-5):
                break
        #"""
        """
        if(i==4):
            if(avg_cost<=-20): #-15 PLV filter #-5 #-90 #-30 #9
                break
        else:
            if(avg_cost<=0): #-15 PLV filter #-5 #-90 #-30 #9
                break
        #"""
        """
        if(prev_cost < avg_cost):
            stop_val += 1
            if (stop_val == stop_num):
                break
        else:
            stop_val = 0
            prev_cost = avg_cost
        """
    Computational_time[i] = Computational_time[i]/num_epoch[i]
    print("computational time for each epoch: ",np.round(Computational_time[i],3))
    if(partial_subject and part_channel):
        test_index = np.where(y_test>=5)[0]
        n_partial = 32
        n = adj_train.shape[1]
        prev_norm = tf.cast((np.mean(adj_train,keepdims=True,axis=0)),tf.float32)
        A_test = np.tile(graph_norm(prev_norm),len(test_index)).reshape(-1,n,n)
        A_test[:,:n_partial,:n_partial] = graph_norm(adj_test[test_index,:n_partial,:n_partial])
        adj_norm_test[test_index] = A_test
    meanr,logvarr = VAEmodel.encoder(tf.cast(features,tf.float32),tf.cast(adj_norm,tf.float32), 0.)
    ztr = VAEmodel.reparameterize(meanr,logvarr)
    meane,logvare = VAEmodel.encoder(tf.cast(features_test,tf.float32),tf.cast(adj_norm_test,tf.float32), 0.)
    zte = VAEmodel.reparameterize(meane,logvare)
    train_feature = deepcopy(ztr).numpy().reshape(len(ztr),-1)
    test_feature = deepcopy(zte).numpy().reshape(len(zte),-1)
    
    Class_method = "SVM"
    svm_prob = False 
    if(Class_method == "KNN"):
        classifier = KNeighborsClassifier(n_neighbors=5)
        classifier.fit(train_feature, y_train)
        t = time.time()
        test_pred = classifier.predict(test_feature)
        print("testing time: ", time.time()-t)
        accuracy[i] = 100 * np.sum(test_pred==(y_test)) / len(test_pred)
        print("accuracy: ", np.round(accuracy[i],3))
#    elif(Class_method == "bayes"):
        mnb = MultinomialNB()
        test_pred = mnb.fit(train_feature-np.min(train_feature), y_train).predict(test_feature-np.min(test_feature))
        accuracy2[i] = 100 * np.sum(test_pred==(y_test)) / len(test_pred)
        print("accuracy: ", np.round(accuracy2[i],3))
    else:
        #SVM better than naive baise 
        clf = svm.SVC(gamma='scale', probability=svm_prob) 
        clf.fit(train_feature,y_train)
        t = time.time()
        test_pred = clf.predict(test_feature)
        print("testing time: ", time.time()-t)
        accuracy[i] = 100 * np.sum(test_pred==(y_test)) / len(test_pred)
        print("accuracy: ", np.round(accuracy[i],3))
    full_time[i] = time.time()-t_start
    print("full time: ",np.round(full_time[i],3))
    if(svm_prob and Class_method=="SVM"):
        test_pred_proba = clf.predict_proba(test_feature)
        eer, _, _, roc = EER_calculation(y_test,test_pred_proba,subject_num)
        EER[i], roc_auc[i] = np.round(np.mean(eer),4),np.round(np.mean(roc),3)
        print("EER: {} and ROC: {}".format(EER[i],roc_auc[i]))
    else:
        eer, _, _, roc = EER_calculation(y_test,test_pred,subject_num)
        EER[i], roc_auc[i] = np.round(np.mean(eer),4),np.round(np.mean(roc),3)
        print("EER: {} and ROC: {}".format(EER[i],roc_auc[i]))

print("final EER: {} and ROC: {}".format(np.round(np.mean(EER),4),np.round(np.mean(roc_auc),3)))
print("final accuracy: ", np.round(np.mean(accuracy),3),np.round(np.var(accuracy),3))
print("final computation time: ",np.round(np.mean(Computational_time),3))
print("final num epochs: ",np.round(np.mean(num_epoch),3))
print("final full time: ",np.round(np.mean(full_time/60),3))

"""
Atr = VAEmodel.decoder(ztr,tf.cast(adj_norm,tf.float32),0.).numpy().reshape(-1,num_nodes,num_nodes) - I
Atr = invlogit(Atr)
Ate = VAEmodel.decoder(zte,tf.cast(adj_norm_test,tf.float32),0.).numpy().reshape(-1,num_nodes,num_nodes) - I_test
Ate = invlogit(Ate)
"""   
"""
#saving and loading the model
VAEmodel.save_weights('./VAE_model/bin10')
VAEmodel = GCNModelVAE(num_features, num_nodes,features_nonzero)
VAEmodel.load_weights('./VAE_model/bin10')

import pickle
pickle.dump(clf, open('./VAE_model/SVM', 'wb'))
clf = pickle.load(open('./VAE_model/SVM', 'rb'))
clf.support_vectors_.shape #space complexity
"""