from scipy.signal import butter, lfilter, filtfilt
from scipy import signal
from sklearn import preprocessing 
import numpy as np
from statsmodels.tsa.stattools import adfuller #for stationary check 
from sklearn.decomposition import FastICA
from skimage import util
from sklearn.utils import shuffle
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a


def bandpass_filter(data, freqband, filtertype, fs, order=5):
    if (freqband == 'delta'):
        lowcut = 0.5
        highcut = 4
    elif (freqband == 'theta'):
        lowcut = 4
        highcut = 8
    elif (freqband == 'alpha'):
        lowcut = 8
        highcut = 14
    elif (freqband == 'beta'):
        lowcut = 14
        highcut = 30
    elif (freqband == 'gamma'):
        lowcut = 30
        highcut = 45
    elif (freqband == 'all'):
        lowcut = .5
        highcut = 45
    if (filtertype == 'butter'):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data)
        #y = lfilter(b, a, data)
    elif (filtertype == 'fir'):
        b = signal.firwin(order,[lowcut, highcut], pass_zero=False, nyq = 0.5*fs)
        y = lfilter(b, [1.0], data)
    return y

def notch_filter(data, fs):
    f0 = 50.0  # Frequency to be removed from signal (Hz)
    Q = 30.0  # Quality factor
    b, a = signal.iirnotch(f0, Q, fs)
    y = lfilter(b, a, data)
    f0 = 60.0  # Frequency to be removed from signal (Hz)
    b, a = signal.iirnotch(f0, Q, fs)
    #y = lfilter(b, a, data)
    y = filtfilt(b, a, y)
    """
    f0 = 120.0  # Frequency to be removed from signal (Hz)
    b, a = signal.iirnotch(f0, Q, fs)
    y = filtfilt(b, a, y)
    #"""
    return y

def adj_matrix(train_features_n, test_features_n, win_size,n_sample_train,n_sample_test, n, A_Matrix='cov'):
    if (A_Matrix=='cov'):
    #covariance matrix 
        x_train_cov = np.einsum('ijk,ilk->ijl',train_features_n,train_features_n)  
        x_train = np.abs(x_train_cov)
        x_test_cov = np.einsum('ijk,ilk->ijl',test_features_n,test_features_n)  
        x_test = np.abs(x_test_cov)
    elif(A_Matrix=='ICA'):
        """
        x_train = []
        for i in range(len(train_features_n)):
            transformer = FastICA(n_components=n,random_state=0, tol=0.0001)
            transformer.fit_transform(train_features_n[i].T)
            x_train.append(transformer.components_)
        """ 
        @ignore_warnings(category=ConvergenceWarning)
        def func(x):
            transformer = FastICA(n_components=n,random_state=0, tol=0.0001)
            transformer.fit_transform(x.T)
            return transformer.components_
        x_train = list(map(func, train_features_n))
        #x_train = [transformer.components_ for i in range(len(train_features_n))]
        x_test = list(map(func, test_features_n))
        x_train, x_test = np.array(x_train), np.array(x_test)
    else:
        #phase matrix
        H_train = signal.hilbert(train_features_n)
        phase_train = (np.angle(H_train))
        H_test = signal.hilbert(test_features_n)
        phase_test = (np.angle(H_test)) #np.unwrap
        if(A_Matrix=='plv' or A_Matrix=='iplv'):
            #PLV_Sample
            x_train_plv = np.einsum('ijk,ilk->ijl',np.exp(phase_train*1j),np.exp(phase_train*-1j))  / (win_size - 1)
            if(A_Matrix=='iplv'):
                x_train = np.abs(x_train_plv.imag)
            else:
                x_train = np.abs(x_train_plv)
            x_test_plv = np.einsum('ijk,ilk->ijl',np.exp(phase_test*1j),np.exp(phase_test*-1j))  / (win_size - 1)
            if(A_Matrix=='iplv'):
                x_test = np.abs(x_test_plv.imag)
            else:
                x_test = np.abs(x_test_plv)
        elif(A_Matrix=='pli'):
            #PLI
            x_train_pli = np.zeros((n_sample_train,n,n))
            for i in range(n):
                x_train_pli[:,i,:] = np.abs(np.mean(np.sign(phase_train[:,i,:].reshape(n_sample_train,1,win_size)-phase_train),axis=2))
            x_train = x_train_pli
            x_test_pli = np.zeros((n_sample_test,n,n))
            for i in range(n):
                x_test_pli[:,i,:] = np.abs(np.mean(np.sign(phase_test[:,i,:].reshape(n_sample_test,1,win_size)-phase_test),axis=2))
            x_test = x_test_pli
        elif(A_Matrix=='AEC'):  
            #AEC
            x_train_aec = np.abs(H_train) - np.mean(np.abs(H_train), axis=2, keepdims=1)
            x_train_aec = x_train_aec / np.sqrt(np.sum(x_train_aec**2, axis=2, keepdims=1)) #normalizing in time
            x_train = np.einsum('ijk,ilk->ijl',x_train_aec,x_train_aec)  / (win_size - 1)
            x_test_aec = np.abs(H_test) - np.mean(np.abs(H_test), axis=2, keepdims=1)
            x_test_aec = x_test_aec / np.sqrt(np.sum(x_test_aec**2, axis=2, keepdims=1)) #normalizing in time
            x_test = np.einsum('ijk,ilk->ijl',x_test_aec,x_test_aec)  / (win_size - 1)    
        else:
            raise Exception("non-existing model")
    return x_train, x_test

def normalizition(train_features,test_features,normalize,n,win_size):
    if(normalize=='maxmin'):
        train_features_n = (train_features - np.min(train_features,axis=2,keepdims=1))/(np.max(train_features,axis=2,keepdims=1)-np.min(train_features,axis=2,keepdims=1))
        train_features_n = 2*train_features_n - 1
        test_features_n = (test_features - np.min(test_features,axis=2,keepdims=1))/(np.max(test_features,axis=2,keepdims=1)-np.min(test_features,axis=2,keepdims=1))
        test_features_n = 2*test_features_n - 1
    elif(normalize=='l1' or normalize=='l2'):
        train_features_n = [preprocessing.normalize(train_features[:,:,i], norm = normalize) for i in range(win_size)]
        train_features_n = np.array(train_features_n).reshape(-1,n,win_size)
        test_features_n = [preprocessing.normalize(test_features[:,:,i], norm = normalize) for i in range(win_size)]
        test_features_n = np.array(test_features_n).reshape(-1,n,win_size)
    elif(normalize=='meanstd'):
        #(x-mean(x))/std(x)
        train_features_n = train_features - np.mean(train_features, axis=2, keepdims=1)
        train_features_n = train_features_n / np.sqrt(np.sum(train_features_n**2,axis=2,keepdims=1))
        test_features_n = test_features - np.mean(test_features, axis = 2, keepdims=1) 
        test_features_n = test_features_n / np.sqrt(np.sum(test_features_n**2,axis=2,keepdims=1))
    else:
        train_features_n = train_features
        test_features_n = test_features
    return train_features_n, test_features_n
    
def preprocess_data(x, Labels, K, Fs, dataset2=False, filt = False, ICA = True, 
                    sh = False, A_Matrix = 'cov', normalize='meanstd',sec=1,
                    percent=.2,sampling=False):
    data_length = x.shape[2]
    n = x.shape[1]
    if(sampling):
        win_size = Fs
        step = Fs//2
    else:
        win_size = Fs*sec
        if(sec>1):
            step = Fs*(sec-1)
        else:
            step = sec*(Fs*0+Fs//2) #1-window*alpha%
    #ratio of number of train test #K-fold validation
    #(int((time/Fs)*0.8))*Fs
    if(dataset2):
        test_index = np.arange(int(.25*K*data_length),int(.25*(K+1)*data_length))
    else:
        test_index = np.arange(int(percent*K*data_length),int(percent*(K+1)*data_length))
    train_index = np.delete(np.arange(data_length),test_index)
    x_train = x[:,:,train_index]
    x_test = x[:,:,test_index]
    if(False): # adding noise
        noise = np.random.normal(0, 1, x_test.shape)
        x_test = x_test+noise
        
    subject_num = x.shape[0]
    #ICA
    if(ICA):
        #if(train_filtered.shape[0]>109):
        if(False):
            x_train = x_train.reshape(109,-1,n,x_train.shape[2])
            x_test = x_test.reshape(109,-1,n,x_test.shape[2])
            X_ICA_train = []
            X_ICA_test = []
            for i in range(109):
                transformer = FastICA(n_components=n,random_state=0, max_iter=200, tol=0.0001) #1000
                X_ICA_train.append(transformer.fit_transform(x_train[i].reshape(-1,n)))
                X_ICA_test.append(transformer.transform(x_test[i].reshape(-1,n)))
            X_ICA_train = np.array(X_ICA_train).reshape(subject_num,n,-1)
            X_ICA_test = np.array(X_ICA_test).reshape(subject_num,n,-1)
        else:
            transformer = FastICA(n_components=n,random_state=0, max_iter=1000, tol=0.0001) #1000
            X_ICA_train = transformer.fit_transform(x_train.reshape(-1,n))
            #transformer.components_
            X_ICA_test = transformer.transform(x_test.reshape(-1,n))                
            X_ICA_train = X_ICA_train.reshape(subject_num,n,-1)
            X_ICA_test = X_ICA_test.reshape(subject_num,n,-1)
    else:
        X_ICA_train = x_train
        X_ICA_test = x_test
        
    if(filt):
        #60Hz filter
        train_filtered = notch_filter(X_ICA_train, Fs)
        test_filtered = notch_filter(X_ICA_test, Fs)
        #band pass filter #gamma, beta, alpha
        #train_filtered = bandpass_filter(train_filtered, 'alpha', 'fir', Fs, 100)
        train_filtered = bandpass_filter(train_filtered, 'beta', 'butter', Fs, 5)
        test_filtered = bandpass_filter(test_filtered, 'beta', 'butter', Fs, 5)
    else:
        #60Hz filter
        """
        train_filtered = notch_filter(X_ICA_train, Fs)
        test_filtered = notch_filter(X_ICA_test, Fs)
        #"""
        """
        train_filtered = bandpass_filter(train_filtered, 'all', 'butter', Fs, 3)
        test_filtered = bandpass_filter(test_filtered, 'all', 'butter', Fs, 3)
        #"""
        #"""
        train_filtered = X_ICA_train
        test_filtered = X_ICA_test
        #"""
    if(dataset2):
        signal.savgol_filter(x, Fs//2, 3)
        
    #windowing data using hamming window
    n_sample_train, _ = util.view_as_windows(x_train[0,0,:], window_shape=(win_size,), step=step).shape
    n_sample_test, _ = util.view_as_windows(x_test[0,0,:], window_shape=(win_size,), step=step).shape
    #fit size of data
    X_ICA_train = X_ICA_train[:,:,:((n_sample_train)*step+win_size-step)]
    X_ICA_test = X_ICA_test[:,:,:((n_sample_test)*step+win_size-step)]
    #win = signal.hamming(win_size)
    win = 1
    
    if(not(dataset2)):
        if(sampling):
            train_features = np.zeros((subject_num,n,win_size,n_sample_train))
            test_features = np.zeros((subject_num,n,win_size,n_sample_test))
            for i in range(0, X_ICA_train.shape[2]-step, step):
                train_features[:,:,:,i//step] = X_ICA_train[:,:,i : i + win_size]
            for i in range(0, X_ICA_test.shape[2]-step, step):
                test_features[:,:,:,i//step] = X_ICA_test[:,:,i : i + win_size]
            len_tr = 200
            len_te = 50
            index_train = np.random.randint(1, high=n_sample_train, size=(len_tr,sec), dtype='l')
            index_test = np.random.randint(1, high=n_sample_test, size=(len_te,sec), dtype='l')
            r_train_features = np.zeros((len_tr,subject_num,n,win_size))
            r_test_features = np.zeros((len_te,subject_num,n,win_size))
            for j in range(len_tr):
                r_train_features[j] = np.mean(train_features[:,:,:,index_train[j]],axis=3)
                if(j<len_te):
                    r_test_features[j] = np.mean(test_features[:,:,:,index_test[j]],axis=3)            
            train_features = r_train_features.reshape(-1,n,win_size)
            test_features = r_test_features.reshape(-1,n,win_size)
            n_sample_train = len_tr
            n_sample_test = len_te
        else:
            train_features = [X_ICA_train[:,:,i : i + win_size]*win for i in range(0, X_ICA_train.shape[2]-step, step)]
            test_features = [X_ICA_test[:,:,i : i + win_size]*win for i in range(0, X_ICA_test.shape[2]-step, step)]
            train_features = np.asarray(train_features).reshape(n_sample_train*subject_num,n,win_size)
            test_features = np.asarray(test_features).reshape(n_sample_test*subject_num,n,win_size)
        
        y_train = np.tile(Labels,n_sample_train)
        y_test = np.tile(Labels,n_sample_test)
        
        n_sample_train = n_sample_train*subject_num
        n_sample_test = n_sample_test*subject_num
        
        #shuffle data 
        if(sh):
            train_features, y_train = shuffle(train_features, y_train)
            test_features, y_test = shuffle(test_features, y_test)
        
        #check whether stationary (p<0.05)
        result = adfuller(train_features[1,1,:])
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical Values:')
        for key, value in result[4].items():
            	print('\t%s: %.3f' % (key, value))
                
        #normalize data 
        #normalize = 'meanstd' 'maxmin' 'l1' 'l2'
        train_features_n, test_features_n = normalizition(train_features,test_features,normalize,n,win_size)
        # create adjency matrix 
        #A_Matrix = 'cov' 'plv' 'iplv' 'pli' 'AEC'
        train_x, test_x = adj_matrix(train_features_n, test_features_n, win_size,n_sample_train,n_sample_test, n, A_Matrix)
    else:
        tr = np.asarray(train_features)
        n_sample_train = n_sample_train//2
        tr1 = tr[:len(tr)//2].reshape(n_sample_train*subject_num,n,win_size)
        tr2 = tr[len(tr)//2:((len(tr)//2)*2)].reshape(n_sample_train*subject_num,n,win_size)
        te = np.asarray(test_features)
        n_sample_test = n_sample_test//2
        te1 = te[:len(te)//2].reshape(n_sample_test*subject_num,n,win_size)
        te2 = te[len(te)//2:((len(te)//2)*2)].reshape(n_sample_test*subject_num,n,win_size)
        
        y_train = np.tile(Labels,n_sample_train)
        y_test = np.tile(Labels,n_sample_test)
        n_sample_train = n_sample_train*subject_num
        n_sample_test = n_sample_test*subject_num
        
        tr1, te1 = normalizition(tr1,te1,normalize,n,win_size)
        tr2, te2 = normalizition(tr2,te2,normalize,n,win_size)
        tr11, te11 = adj_matrix(tr1, te1, win_size,n_sample_train,n_sample_test, n, A_Matrix)
        tr22, te22 = adj_matrix(tr2, te2, win_size,n_sample_train,n_sample_test, n, A_Matrix)
        tr12 = np.abs(np.einsum('ijk,ilk->ijl',tr1,tr2)) 
        tr21 = np.abs(np.einsum('ijk,ilk->ijl',tr2,tr1)) 
        train_x = np.concatenate((np.concatenate((tr11,tr12),axis=2),np.concatenate((tr21,tr22),axis=2)),axis=1)
        te12 = np.abs(np.einsum('ijk,ilk->ijl',te1,te2)) 
        te21 = np.abs(np.einsum('ijk,ilk->ijl',te2,te1)) 
        test_x = np.concatenate((np.concatenate((te11,te12),axis=2),np.concatenate((te21,te22),axis=2)),axis=1)
        
    return train_x, test_x, y_train, y_test

def preprocess_data_task(x, Fs, ratio, filt = False, ICA = True, 
                    sh = False, A_Matrix = 'cov', normalize='meanstd'):
    data_length = x.shape[3]
    n = x.shape[2]
    win_size = Fs
    step = Fs*0+Fs//2 #1-window*alpha%
    num_train = int(np.ceil(len(x)*ratio))
    x_train = x[:num_train,:]
    x_test = x[num_train:,:]
    if(x.shape[1]==14):
        Labels = np.concatenate((np.concatenate((np.arange(6),np.arange(2,6))),np.arange(2,6))) + 1
    else:
        Labels = np.arange(x.shape[1]) + 1
    Lables_train = np.tile(Labels,x_train.shape[0])
    Lables_test = np.tile(Labels,x_test.shape[0])
    x_train = x_train.reshape(-1,n,data_length)
    x_test = x_test.reshape(-1,n,data_length)
    
    if(filt):
        #60Hz filter
        train_filtered = notch_filter(x_train, Fs)
        test_filtered = notch_filter(x_test, Fs)
        #band pass filter
        #train_filtered = bandpass_filter(train_filtered, 'alpha', 'fir', Fs, 100)
        train_filtered = bandpass_filter(train_filtered, 'beta', 'butter', Fs, 5)
        test_filtered = bandpass_filter(test_filtered, 'beta', 'butter', Fs, 5)
    else:
        #60Hz filter
        """
        train_filtered = notch_filter(x_train, Fs)
        test_filtered = notch_filter(x_test, Fs)
        #"""
        train_filtered = x_train
        test_filtered = x_test
    
    #ICA
    if(ICA):
        transformer = FastICA(n_components=n,random_state=0, max_iter=200, tol=0.0001) #1000
        X_ICA_train = transformer.fit_transform(train_filtered.reshape(-1,n))
        X_ICA_train = X_ICA_train.reshape(-1,n,data_length)
        X_ICA_test = transformer.transform(test_filtered.reshape(-1,n))
        X_ICA_test = X_ICA_test.reshape(-1,n,data_length)
    else:
        X_ICA_train = train_filtered
        X_ICA_test = test_filtered
        
    #windowing data using hamming window
    n_sample_train, _ = util.view_as_windows(x_train[0,0,:], window_shape=(win_size,), step=step).shape
    n_sample_test, _ = util.view_as_windows(x_test[0,0,:], window_shape=(win_size,), step=step).shape
    
    #win = signal.hamming(win_size)
    win = 1
        
    train_features = [X_ICA_train[:,:,i : i + win_size]*win for i in range(0, x_train.shape[2]-step, step)]
    train_features = np.asarray(train_features).reshape(-1,n,win_size)
    
    test_features = [X_ICA_test[:,:,i : i + win_size]*win for i in range(0, x_test.shape[2]-step, step)]
    test_features = np.asarray(test_features).reshape(-1,n,win_size)
    
    y_train = np.tile(Lables_train,n_sample_train)
    y_test = np.tile(Lables_test,n_sample_test)
    
    n_sample_train = train_features.shape[0]
    n_sample_test = test_features.shape[0]
    
    #shuffle data 
    if(sh):
        train_features, y_train = shuffle(train_features, y_train)
        test_features, y_test = shuffle(test_features, y_test)
    
    #check whether stationary (p<0.05)
    result = adfuller(train_features[1,1,:])
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        	print('\t%s: %.3f' % (key, value))
            
    #normalize data 
    #normalize = 'meanstd' 'maxmin' 'l1' 'l2'
    train_features_n, test_features_n = normalizition(train_features,test_features,normalize,n,win_size)
    # create adjency matrix 
    #A_Matrix = 'cov' 'plv' 'iplv' 'pli' 'AEC'
    train_x, test_x = adj_matrix(train_features_n, test_features_n, win_size,n_sample_train,n_sample_test, n, A_Matrix)
    
    return train_x, test_x, y_train, y_test

def preprocess_data_BCI(x_train,x_test, Labels, Fs, filt = False, ICA = True, 
                     A_Matrix = 'cov', normalize='meanstd',sec=1,sampling=False):
    n = x_train.shape[1]
    if(sampling):
        win_size = Fs
        step = Fs//2
    else:
        win_size = Fs*sec
        step = sec*(Fs*0+Fs//2) 
    subject_num = x_train.shape[0]

    if(filt):
        #60Hz filter
        train_filtered = notch_filter(x_train, Fs)
        test_filtered = notch_filter(x_test, Fs)
        #band pass filter
        #train_filtered = bandpass_filter(train_filtered, 'alpha', 'fir', Fs, 100)
        train_filtered = bandpass_filter(train_filtered, 'gamma', 'butter', Fs, 5)
        test_filtered = bandpass_filter(test_filtered, 'gamma', 'butter', Fs, 5)
    else:
        #60Hz filter
        train_filtered = notch_filter(x_train, Fs)
        test_filtered = notch_filter(x_test, Fs)
        #train_filtered = x_train
        #test_filtered = x_test
    #ICA
    if(ICA):
        transformer = FastICA(n_components=n,random_state=0, max_iter=1000, tol=0.0001) #1000
        X_ICA_train = transformer.fit_transform(train_filtered.reshape(-1,n))
        X_ICA_train = X_ICA_train.reshape(subject_num,n,-1)
        X_ICA_test = transformer.transform(test_filtered.reshape(-1,n))
        X_ICA_test = X_ICA_test.reshape(subject_num,n,-1)
    else:
        X_ICA_train = train_filtered
        X_ICA_test = test_filtered
        
    #windowing data using hamming window
    n_sample_train, _ = util.view_as_windows(x_train[0,0,:], window_shape=(win_size,), step=step).shape
    n_sample_test, _ = util.view_as_windows(x_test[0,0,:], window_shape=(win_size,), step=step).shape
    
    X_ICA_train = X_ICA_train[:,:,:((n_sample_train)*step+win_size-step)]
    X_ICA_test = X_ICA_test[:,:,:((n_sample_test)*step+win_size-step)]
    #win = signal.hamming(win_size)
    win = 1
    if(sampling):
        train_features = np.zeros((subject_num,n,win_size,n_sample_train))
        test_features = np.zeros((subject_num,n,win_size,n_sample_test))
        for i in range(0, X_ICA_train.shape[2]-step, step):
            train_features[:,:,:,i//step] = X_ICA_train[:,:,i : i + win_size]
        for i in range(0, X_ICA_test.shape[2]-step, step):
            test_features[:,:,:,i//step] = X_ICA_test[:,:,i : i + win_size]
        len_tr = 200
        len_te = 50
        index_train = np.random.randint(1, high=n_sample_train, size=(len_tr,sec), dtype='l')
        index_test = np.random.randint(1, high=n_sample_test, size=(len_te,sec), dtype='l')
        r_train_features = np.zeros((len_tr,subject_num,n,win_size))
        r_test_features = np.zeros((len_te,subject_num,n,win_size))
        for j in range(len_tr):
            r_train_features[j] = np.mean(train_features[:,:,:,index_train[j]],axis=3)
            if(j<len_te):
                r_test_features[j] = np.mean(test_features[:,:,:,index_test[j]],axis=3)            
        train_features = r_train_features.reshape(-1,n,win_size)
        test_features = r_test_features.reshape(-1,n,win_size)
        n_sample_train = len_tr
        n_sample_test = len_te
    else:
        train_features = [X_ICA_train[:,:,i : i + win_size]*win for i in range(0, X_ICA_train.shape[2]-step, step)]
        test_features = [X_ICA_test[:,:,i : i + win_size]*win for i in range(0, X_ICA_test.shape[2]-step, step)]
        
        train_features = np.asarray(train_features).reshape(n_sample_train*subject_num,n,win_size)
        test_features = np.asarray(test_features).reshape(n_sample_test*subject_num,n,win_size)
    
    y_train = np.tile(Labels,n_sample_train)
    y_test = np.tile(Labels,n_sample_test)
    
    n_sample_train = n_sample_train*subject_num
    n_sample_test = n_sample_test*subject_num
    
    #check whether stationary (p<0.05)
    result = adfuller(train_features[1,1,:])
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        	print('\t%s: %.3f' % (key, value))

    train_features_n, test_features_n = normalizition(train_features,test_features,normalize,n,win_size)
    train_x, test_x = adj_matrix(train_features_n, test_features_n, win_size,n_sample_train,n_sample_test, n, A_Matrix)
    return train_x, test_x, y_train, y_test

def dataset2_indices(signal_channel):
    channel7_index = np.zeros((9),dtype=int)
    channel7_index[0] = signal_channel.index('Fz..')
    channel7_index[1] = signal_channel.index('Cz..')
    channel7_index[2] = signal_channel.index('T7..') #T3
    channel7_index[3] = signal_channel.index('T8..') #T4
    channel7_index[4] = signal_channel.index('C3..')
    channel7_index[5] = signal_channel.index('C4..')
    channel7_index[6] = signal_channel.index('Oz..')
    channel7_index[7] = signal_channel.index('Fp1.')
    channel7_index[8] = signal_channel.index('Fp2.')
    return np.sort(channel7_index)

    