import numpy as np
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
import matplotlib.pyplot as plt

def EER_calculation(y_test,test_pred,subject_num):
    y = tf.keras.utils.to_categorical(y_test-1, num_classes=subject_num)
    if(test_pred.ndim==1):
        y_pred = tf.keras.utils.to_categorical(test_pred-1, num_classes=subject_num)
    else:
        y_pred = test_pred
    fpr = dict()
    tpr = dict()
    roc_auc = np.zeros(subject_num)
    eer = np.zeros(subject_num)
    for i in range(subject_num):
        fpr[i], tpr[i], _ = roc_curve(y[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        fnr = 1 - tpr[i]
        eer[i] = fnr[np.nanargmin(np.absolute((fnr - fpr[i])))]
    return eer, fpr, tpr, roc_auc