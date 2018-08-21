import numpy
from scipy.io import loadmat
import tensorflow as tf
from sklearn.preprocessing import normalize, scale
from sklearn.metrics import *
from sklearn.preprocessing import minmax_scale
from datetime import datetime
import pandas as pd
import shutil
import numpy as np
import json
import sys
import os
from sklearn import svm
from sklearn import linear_model
from keras.models import load_model
from keras import backend as K
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

sys.path.append('/home/alexanderliao/data/GitHub/')

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score

if False:
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
	os.environ["CUDA_VISIBLE_DEVICES"] = ""

def evaluate_model():
	acc = accuracy_score(Ytest,clf.predict(Xtest))
	print("Accuracy: {}".format(acc))
	f1 = f1_score(Ytest,clf.predict(Xtest))
	print("F1 Score: {}".format(f1))
	jaccard = jaccard_similarity_score(Ytest,clf.predict(Xtest))
	print("Jaccard Similarity Coefficient : {}".format(jaccard))
	zero_one = zero_one_loss(Ytest,clf.predict(Xtest))
	print("Zero-One Loss : {}".format(zero_one))
	

print("Reading training data...")
Xtrain=pd.read_csv('data/conf_feature_train.csv' ,dtype='double')
Xtrain=scale(Xtrain.dropna(axis=1).loc[:, ~(Xtrain == 0).any(0)])

print(" ")
print("Sanity Check:")
print(" ")
print(Xtrain)
Ytrain=pd.read_csv('data/conf_label_train.csv' ,dtype='double').dropna(axis=1).values.reshape(-1)
print(" ")
print(Ytrain)
print("Done!")
print(" ")

print("Reading test data...")
Xtest=pd.read_csv('data/conf_feature_test.csv' ,dtype='double')
Xtest=scale(Xtest.dropna(axis=1).loc[:, ~(Xtest == 0).any(0)])

Ytest=pd.read_csv('data/conf_label_test.csv' ,dtype='double').dropna(axis=1).values.reshape(-1)
print("Done!")
print(" ")

numpy.random.seed(7)

print("Dataset Shape: (X;Y)")
print(Xtrain.shape)
print(Ytrain.shape)
print(" ")
print("Seed: {}".format(np.random.get_state()[1][0]))

#k=50000
#print("Selecting {} samples from training set.".format(k))
#Xtrain=Xtrain[0:k-1,:]
#Ytrain=Ytrain[0:k-1]
print(" ")


print("====================================")
print("Confidence Only: (Classification)")
print("====================================")

print(" ")

print("-------------------------------------")
print("Neural Network: (Threshold 0.5)")
print(" ")
print(" ")
clf = load_model("conf_baseline.model")
#evaluate_model()
result = clf.predict(Xtest).reshape(-1)
result[result>0.5] = 1
result[result<=0.5] = 0

acc = accuracy_score(Ytest,result)
print(" ")
print(" ")
print("Accuracy: {}".format(acc))
f1 = f1_score(Ytest,result)
print("F1 Score: {}".format(f1))
jaccard = jaccard_similarity_score(Ytest,result)
print("Jaccard similarity coefficient : {}".format(jaccard))
zero_one = zero_one_loss(Ytest,result)
print("Zero-One Loss : {}".format(zero_one))

print("-------------------------------------")
print("Vanilla Support Vector Machine:")
clf = svm.SVC(n_jobs=4,verbose=True)
clf.fit(Xtrain,Ytrain)
evaluate_model()


print("-------------------------------------")
print("Vanilla Logistic Regression")
clf = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6,verbose=1,n_jobs=4,solver='lbfgs')
clf.fit(Xtrain, Ytrain)
evaluate_model()

print("-------------------------------------")
print("Vanilla Bayesian Ridge Regression")
clf = linear_model.BayesianRidge(verbose=True)
clf.fit(Xtrain, Ytrain)
evaluate_model()


print("-------------------------------------")
print("Vanilla Decision Tree")
clf = tree.DecisionTreeClassifier(n_jobs=4)
clf.fit(Xtrain, Ytrain)
evaluate_model()

print("-------------------------------------")
print("Vanilla Random Forest")
clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=200)
clf.fit(Xtrain, Ytrain)
evaluate_model()

print("-------------------------------------")
print("AdaBoosted Decision Free")
clf = RandomForestClassifier(n_jobs=4)
clf.fit(Xtrain, Ytrain)
evaluate_model()

print("====================================")
print("Estimation Only: (Regression)")
print("====================================")
