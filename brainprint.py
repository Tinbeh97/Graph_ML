import numpy as np 
import networkx.algorithms as nl
from graphfeatures import matrix_feature
from sklearn.metrics import confusion_matrix
from copy import deepcopy
from sklearn import svm
import time 
from pre_func import dataset2_indices, preprocess_data
from evaluation import EER_calculation

def invlogit(z):
    return 1 - 1 /(1 + np.exp(z))

def Maha_dist(test_f, train_f):
    #train_f = train_f-np.mean(train_f,axis=1,keepdims=1)
    C = np.cov(train_f.T)
    C_inv = np.linalg.pinv(C)
    u = np.mean(train_f,axis=0,keepdims=1)
    v = test_f - np.mean(test_f,axis=1,keepdims=1)
    D = np.dot(np.dot((v-u) , C_inv) , (v-u).T)
    return D.diagonal()

connected = 1
VAE = False

nb_run = 5
accuracy = np.zeros((nb_run,1))
Computational_time = np.zeros((nb_run,1))
full_time = np.zeros((nb_run,1))
roc_auc = np.zeros((nb_run,1))
EER = np.zeros((nb_run,1))

for i in range(nb_run):
    t_begin = time.time()
    train_x, test_x, y_train, y_test = preprocess_data(physio_data[:,:n,:],Labels,i,
                                                       Fs,dataset2=False,filt=False,
                                                       ICA=True,A_Matrix='plv',sec=2)
    n = train_x.shape[2]
    print('training features')
    t_start = time.time()
    train_feature = matrix_feature(train_x,n,connected)
    Computational_time[i] = time.time()-t_start
    print("feature extraction time: ", Computational_time[i])
    
    print('testing features')
    test_feature = matrix_feature(test_x,n,connected)
    
    discriminator = 'Maha_dist'
    if(discriminator=='Maha_dist'):
        sort_index = np.argsort(y_train)
        train_f_sort = train_feature[sort_index,:]
        num = len(sort_index)//subject_num
        pred = np.zeros((test_feature.shape[0],subject_num))
        
        for j in range(subject_num):
            pred[:,j] = Maha_dist(test_feature,train_f_sort[j*num:(j+1)*num,:])
        
        test_pred = np.argmin(pred,axis=1)
        accuracy[i] = 100 * np.sum(test_pred==(y_test-1)) / len(test_pred)
        c = confusion_matrix(y_test-1, test_pred)
    elif(discriminator=='SVM'):
        #clf = svm.SVC(probability=True,gamma='scale')
        clf = svm.SVC(gamma='scale')
        clf.fit(train_feature,y_train)
        test_pred = clf.predict(test_feature)
        accuracy = 100 * np.sum(test_pred==(y_test)) / len(test_pred)
        c = confusion_matrix(y_test, test_pred)
    else:
        raise Exception("non-existing model")
    
    
    print("accuracy: ", accuracy[i])
    crr = np.sum(c.diagonal())/(np.sum(c))
    print("CRR: ", crr)
    full_time[i] = time.time()-t_begin
    print("whole time: ",full_time[i])
    eer, _, _, roc = EER_calculation((y_test-1),test_pred,subject_num)
    EER[i], roc_auc[i] = np.round(np.mean(eer),4),np.round(np.mean(roc),3)
    print("EER: {} and ROC: {}".format(EER[i],roc_auc[i]))

print("final EER: {} and ROC: {}".format(np.round(np.mean(EER),4),np.round(np.mean(roc_auc),3)))
print("final accuracy: ", np.round(np.mean(accuracy),3),np.round(np.var(accuracy),3))
print("final computation time: ",np.round(np.mean(Computational_time),3))
print("final full time: ",np.round(np.mean(full_time/60),3))
