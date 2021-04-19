from __future__ import division
from __future__ import print_function
from graphfeatures import graph_norm
import numpy as np
import tensorflow as tf
import time
from copy import deepcopy
from Graph_layers import GraphConvolution, Graph_diffpool,Graph_sagepool, Graph_globalpool, Graph_clustpool
from Graph_layers import Graph_clustpool_2
from clustering import A_binarize, creating_label
from evaluation import EER_calculation
#from keras.utils.vis_utils import plot_model

diffpool = False 
sagepool = False
globalpool = False
mypool = False 
Task = False; ntask = 6
def invlogit(z):
    return 1 - 1 /(1 + np.exp(z))
    
class GCNModel(tf.keras.Model):
    def __init__(self, adj,adj_norm, num_features, num_nodes, features_nonzero, subject_num, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.hd1 = 
        self.hd2 = 
        self.hd3 = 
        self.subject_num = subject_num if(not(Task)) else ntask
        self.h1 = GraphConvolution(input_dim = self.input_dim, 
                                   output_dim = self.hd1, num = 1,
                                   act = tf.nn.leaky_relu)
        """
        self.h2 = GraphConvolution(input_dim = self.hd1,
                                   output_dim = self.hd2, num = 2,
                                   act = lambda x: x)
        #"""
        """
        self.h5 = GraphConvolution(input_dim = self.hd2,
                                   output_dim = self.hd2, num = 5,
                                   act = lambda x: x) 
        #"""
        #"""
        self.h3 = GraphConvolution(input_dim = self.hd1,
                                   output_dim = self.hd3, num = 3,
                                   act = tf.nn.tanh)  #leaky_relu
        #"""
        self.h4 = tf.keras.layers.Dense(self.subject_num)
        if(diffpool):
            self.p1 = Graph_diffpool(input_dim = self.hd1,
                                     output_dim = 48, num = 4,
                                     act = lambda x: x)
        elif(sagepool):
            self.p1 = Graph_sagepool(input_dim = self.hd1, num = 4, ratio = .25,
                                     act = lambda x: x)
        if(mypool):
            #self.p1 = Graph_clustpool_2(adj,ratio=.25)
            self.p1 = Graph_clustpool(adj,48,cluster_type='sum')
            self.adj_pool = self.p1.adj_masking(adj_norm)
            #self.adj_pool = tf.matmul(adj_pool, adj_pool)
            """
            feature = tf.ones((adj.shape[0],adj.shape[1],1))
            x, adj2 = self.p1(feature,adj)
            self.p2 = Graph_clustpool_2(adj2,ratio=.25)
            #"""
        
    def call(self, inputs, adj, rate, adj_pool):
        adj = tf.matmul(adj, adj)
        x = self.h1(inputs, adj, rate)
        if(mypool):
            x, _ = self.p1(x, adj)
        #x = self.h2(x, adj, rate)
        #x = self.h5(x, adj, rate)
        if(diffpool):
            x, adj = self.p1(x, adj, rate)
        elif(sagepool):
            x, adj = self.p1(x, adj, rate)
        #x = self.h2(x, adj, rate)
        """
        if(sagepool):
            x, adj = self.p2(x, adj, rate, .25)
        """
        """
        if(mypool):
            x, adj = self.p2(x, adj)
        #"""
        if(adj_pool==None):
            adj_pool = adj
        x = self.h3(x, adj_pool, rate)
            
        if(globalpool):  
            x = Graph_globalpool(pool_method='max')(x)
        else:
            x = tf.keras.layers.Flatten()(x)
        x = self.h4(x)
        x = tf.nn.log_softmax(x, axis=1)
        return x

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=.4e-2,
                decay_steps=10000,
                decay_rate=0.9)
#Model Optimizer
class Optimizer(object):
    def __init__(self, subject_num):
        self.cce = tf.keras.losses.CategoricalCrossentropy()
        self.subject_num = subject_num if(not(Task)) else ntask
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule) #RMSprop
    def train_step(self,y,x,adj,rate,adj2,model):
        with tf.GradientTape() as tape: #watch_accessed_variables=False
            tape.watch(model.trainable_variables)
            y_pred = model(x,adj,rate,adj2)
            y_true = tf.keras.utils.to_categorical(y-1, num_classes=self.subject_num)
            #loss = self.cce(y_true, y_pred)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred,labels=y_true))
            if(diffpool):
                loss += sum(model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        opt_op = self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss
    
dataset7 = True
if(dataset7):
    Binary=False
else:
    Binary=True
Part_channel = False #Consider part of the channels

def Adj_matrix(train_x, test_x):   
    if(Binary):
        percentile = 0.9 #0.9
        adj_train = A_binarize(A_matrix=train_x,percent=percentile,sparse=True)
        adj_test  = A_binarize(A_matrix=test_x,percent=percentile,sparse=True)
    else:
        adj_train = deepcopy(train_x) 
        adj_test = deepcopy(test_x)
    if(Part_channel):
        index = creating_label(ztr,y_train,subject_num,method='mean_sort') #dataset2_indices(signal_channel)
        adj_train = adj_train[:,:,index]
        adj_train = adj_train[:,index]
        adj_test = adj_test[:,:,index]
        adj_test = adj_test[:,index]
    return adj_train, adj_test

FLAGS_features = False
if not FLAGS_features:
    features_init_train = None
else:
    features_init_train = deepcopy(ztr)
if not FLAGS_features:
    features_init_test = None
else:
    features_init_test = deepcopy(zte)

verbose = True
nb_run = 5
accuracy = np.zeros((nb_run,1))
Computational_time = np.zeros((nb_run,1))
roc_auc = np.zeros((nb_run,1))
EER = np.zeros((nb_run,1))
num_epoch = np.zeros((nb_run,1))
full_time = np.zeros((nb_run,1))

for i in range(nb_run):
    t_start = time.time()
    subject_num = len(Labels)    
    if(not(dataset7)):
        train_x, test_x, y_train, y_test = preprocess_data(x_original_all[:,0],Labels,i,Fs,dataset2=False,
                                                           filt=False,ICA=True,A_Matrix='cov')
    else:
        train_x, test_x, y_train, y_test = preprocess_data(x_original[:,:,Fs*9:],Labels,i,Fs,
                                                          dataset2=False,filt=False,ICA=True,A_Matrix='plv',sec=30,sampling=False)
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
    if(dataset7):
        adj_norm = adj_train
        adj_norm_test = adj_test
    else:
        adj_norm = graph_norm(adj_train)
        adj_norm_test = graph_norm(adj_test)
    #adj_norm = A[:len(adj_train)]
    #adj_norm_test = A[len(adj_train):]
    
    if not FLAGS_features:
        features_test = np.ones((adj_test.shape[0],adj_test.shape[1],1))
        #features_test = deepcopy(I_test)
    else:
        features_test = deepcopy(features_init_test)
        
    
    rate_test = 0
    #model
    GCmodel = GCNModel(adj_norm,adj_norm,num_features,num_nodes,features_nonzero,subject_num)
    if(mypool):
        print('number of cluster: ',GCmodel.p1.n_cluster)
        adj_pool = GCmodel.adj_pool
        train_dataset = (tf.data.Dataset.from_tensor_slices((adj_norm,y_train,features,adj_pool))
                         .shuffle(len(adj_norm)).batch(64))
    else:
        train_dataset = (tf.data.Dataset.from_tensor_slices((adj_norm,y_train,features))
                         .shuffle(len(adj_norm)).batch(64))
    # Optimizer
    opt = Optimizer(subject_num)
    # Model training
    if verbose:
        print("Training...")
    prev_cost = 100000
    stop_val = 0
    stop_num = 15 #15
    FLAGS_shuffle = False
    nb_epochs = 50
    if(i==0):
        nb_epochs = 40 #80
    for epoch in range(nb_epochs):
        num_epoch[i] +=1
        t = time.time()
        # Compute average loss
        loss = 0
        if(mypool):
            for adj, label, x, adj2 in train_dataset:
                loss += opt.train_step(tf.cast(label,tf.float32),tf.cast(x,tf.float32),
                                       tf.cast(adj,tf.float32), 0.5, adj2, GCmodel)
        else:
            for adj, label, x in train_dataset:
                loss += opt.train_step(tf.cast(label,tf.float32),tf.cast(x,tf.float32),
                                       tf.cast(adj,tf.float32), 0.5, None, GCmodel)
        #loss = opt.train_step(adj_label,tf.cast(features,tf.float32),tf.cast(adj_norm,tf.float32), 0.5, model)
        avg_cost = loss.numpy()
        Computational_time[i] += (time.time() - t)
        if verbose:
            # Display epoch information
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
                  "time=", "{:.5f}".format(time.time() - t))
        nb_epochs += 1
        if(prev_cost <= avg_cost):
            stop_val += 1
            if (stop_val == stop_num):
                break
        else:
            stop_val = 0
            prev_cost = avg_cost
    if(mypool):
        pred = GCmodel(tf.cast(features_test,tf.float32), tf.cast(adj_norm_test,tf.float32), 
                       0.0,GCmodel.p1.adj_masking(adj_norm_test)).numpy()
    else:
        pred = GCmodel(tf.cast(features_test,tf.float32), tf.cast(adj_norm_test,tf.float32), 
                       0.0,None).numpy()
    test_pred = np.argmax(pred,axis=1)
    full_time[i] = time.time()-t_start
    accuracy[i] = 100 * np.sum(test_pred==(y_test-1)) / len(test_pred)
    print("accuracy: ", accuracy[i])
    Computational_time[i] = Computational_time[i]/nb_epochs
    print("computational time for each epoch: ",Computational_time[i])
    eer_num = subject_num if(not(Task)) else ntask
    eer, _, _, roc = EER_calculation(y_test,test_pred+1,eer_num)
    EER[i], roc_auc[i] = np.round(np.mean(eer),4),np.round(np.mean(roc),3)
    print("EER: {} and ROC: {}".format(EER[i],roc_auc[i]))

print("final EER: {} and ROC: {}".format(np.round(np.mean(EER),4),np.round(np.mean(roc_auc),3)))
print("final accuracy: ", np.round(np.mean(accuracy),3),np.round(np.var(accuracy),3))
print("final computation time: ",np.round(np.mean(Computational_time),3))
print("final num epochs: ",np.round(np.mean(num_epoch),3))
print("final full time: ",np.round(np.mean(full_time/60),3))
