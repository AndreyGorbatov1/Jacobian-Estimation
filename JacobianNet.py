from keras.layers import Input, Dense, Dropout, BatchNormalization, PReLU, PReLU
from keras.models import Model, Sequential
from keras import optimizers
from keras import regularizers
import numpy
from scipy.io import loadmat
import tensorflow as tf
from sklearn.preprocessing import normalize, scale
from sklearn.metrics import r2_score
from sklearn.preprocessing import minmax_scale
from datetime import datetime
import pandas as pd
import shutil
import keras
import json
import sys
import os
from sklearn.metrics import r2_score
#a=pd.read_csv('featureTrain.csv' ,dtype='double')
#print(a)

import tensorflow as tf
def huber_loss(y_true, y_pred):
    return tf.losses.huber_loss(y_true,y_pred)

sys.path.append('/home/alexanderliao/data/GitHub/')
from kerasresnet import resnet

"""
Xtrain=scale(pd.read_csv('featureTrain.csv' ,dtype='double').dropna(axis=1),axis=0)
Ytrain=pd.read_csv('labelTrain.csv' ,dtype='double').dropna(axis=1)

Xtest=scale(pd.read_csv('featureTest.csv' ,dtype='double').dropna(axis=1),axis=0)
Ytest=pd.read_csv('labelTest.csv' ,dtype='double').dropna(axis=1)
"""


print("Reading training data...")
Xtrain=pd.read_csv('data/jacob_feature_train.csv' ,dtype='double')
Xtrain=scale(Xtrain.dropna(axis=1).loc[:, ~(Xtrain == 0).any(0)])

#Xtrain=Xtrain[0:k,:]

#Xtrain=Xtrain.reshape((-1,1,len(Xtrain[0,:]),1))

#Xtrain=Xtrain.reshape((k,2,-1,1))

print(" ")
print("Sanity Check:")
print(Xtrain)
Ytrain=pd.read_csv('data/jacob_label_train.csv' ,dtype='double').dropna(axis=1)
Ytrain=minmax_scale(Ytrain.loc[:, ~(Ytrain == 0).any(0)],feature_range=(-1,1))

print("Done!")
#Ytrain=Ytrain.values.reshape((-1,1))

#Ytrain=Ytrain[0:k,:]


print("Reading test data...")
Xtest=pd.read_csv('data/jacob_feature_test.csv' ,dtype='double')
Xtest=scale(Xtest.dropna(axis=1).loc[:, ~(Xtest == 0).any(0)])


#Xtest=Xtest.reshape((-1,1,len(Xtest[0,:]),1))


#Xtest=Xtest.reshape((300,2,-1,1))

Ytest=pd.read_csv('data/jacob_label_test.csv' ,dtype='double').dropna(axis=1)
Ytest=minmax_scale(Ytest.loc[:, ~(Ytest == 0).any(0)],feature_range=(-1,1))
print("Done!")
#Ytest=Ytest.values.reshape((-1,1))

#Ytest=Ytest[0:k,:]

# fix random seed for reproducibility
numpy.random.seed(7)