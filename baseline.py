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
from sklearn import svm

sys.path.append('/home/alexanderliao/data/GitHub/')
from kerasresnet import resnet

print("Reading training data...")
Xtrain=pd.read_csv('data/conf_feature_train.csv' ,dtype='double')
Xtrain=scale(Xtrain.dropna(axis=1).loc[:, ~(Xtrain == 0).any(0)])

print(" ")
print("Sanity Check:")
print(Xtrain)
Ytrain=pd.read_csv('data/conf_label_train.csv' ,dtype='double').dropna(axis=1)
print("Done!")

print("Reading test data...")
Xtest=pd.read_csv('data/conf_feature_test.csv' ,dtype='double')
Xtest=scale(Xtest.dropna(axis=1).loc[:, ~(Xtest == 0).any(0)])

Ytest=pd.read_csv('data/conf_label_test.csv' ,dtype='double').dropna(axis=1)
print("Done!")

numpy.random.seed(7)
#print("Seed: {}".format(numpy.random.get_state()))

print(Ytrain.shape())

print("Support Vector Machine:")
clf = svm.SVC(verbose=True)
clf.fit(Xtrain,Ytrain)

acc=r2_score(Ytest,clf.predict(Xtest))
print(acc)


def routine(Ytest,nn_predictor):
    acc=r2_score(Ytest,nn_predictor.predict(Xtest))
    print(acc)
    #string=str(datetime.now()).replace(".","").replace(" ","")+'-'+str(round(acc,2))
    
    print(r2_score(Ytest,nn_predictor.predict(Xtest)))
    return string
