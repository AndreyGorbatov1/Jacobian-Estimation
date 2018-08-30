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
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

import xgboost as xgb
import lightgbm as lgbm
import time
import pickle 
from estimation_only import nn_1


sys.path.append('/mydisk/Programming/Git/')

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

def class_evaluate_model():
	start = time.time()
	result = clf.predict(Xtest)
	end = time.time()

	elapsed = round(float(end-start),4)
	length = len(Xtest)
	avg = round(elapsed/length,4)
	print("Time taken: {} seconds for evaluation on {} samples; on average: {}".format(elapsed,length,avg))
	
	acc = accuracy_score(Ytest,result).round(4)
	print("Accuracy: {}".format(acc))
	f1 = f1_score(Ytest,result).round(4)
	print("F1 Score: {}".format(f1))
	jaccard = jaccard_similarity_score(Ytest,result).round(4)
	print("Jaccard Similarity Coefficient : {}".format(jaccard))
	zero_one = zero_one_loss(Ytest,result).round(4)
	print("Zero-One Loss : {}".format(zero_one))
	prec = precision_score(Ytest,result).round(4)
	print("Precision : {}".format(prec))
	rec = recall_score(Ytest,result).round(4)
	print("Recall : {}".format(rec))
	matt = matthews_corrcoef(Ytest,result).round(4)
	print("Matthews Correlation Coefficient (MCC) : {}".format(matt))

def reg_evaluate_model():
	start = time.time()
	result=clf.predict(Xtest)
	end = time.time()

	elapsed = round(float(end-start),4)
	length = len(Xtest)
	avg = round(elapsed/length,4)
	print("Time taken: {} seconds for evaluation on {} samples; on average: {}".format(elapsed,length,avg))
	
	mae = mean_absolute_error(Ytest,result).round(4)
	print("Mean Absolute Error: {}".format(mae))

	mse = mean_squared_error(Ytest,result).round(4)
	print("Mean Squared Error: {}".format(mse))

	evs = explained_variance_score(Ytest,result).round(4)
	print("Explained Variance Score: {}".format(evs))

	r2 = r2_score(Ytest,result).round(4)
	print("R2 Score: {}".format(r2))
	
def class_evaluate_model_threshold():
	start = time.time()

	result = clf.predict(Xtest).reshape(-1)
	result[result>0.5] = 1
	result[result<=0.5] = 0

	end = time.time()
	elapsed = round(float(end-start),4)
	length = len(Xtest)
	avg = round(elapsed/length,4)

	print("Time taken: {} seconds for evaluation on {} samples; on average: {}".format(elapsed,length,avg))

	acc = accuracy_score(Ytest,result).round(4)
	print("Accuracy: {}".format(acc))
	f1 = f1_score(Ytest,result).round(4)
	print("F1 Score: {}".format(f1))
	jaccard = jaccard_similarity_score(Ytest,result).round(4)
	print("Jaccard similarity coefficient : {}".format(jaccard))
	zero_one = zero_one_loss(Ytest,result).round(4)
	print("Zero-One Loss : {}".format(zero_one))
	prec = precision_score(Ytest,result).round(4)
	print("Precision : {}".format(prec))
	rec = recall_score(Ytest,result).round(4)
	print("Recall : {}".format(rec))
	matt = matthews_corrcoef(Ytest,result).round(4)
	print("Matthews Correlation Coefficient (MCC) : {}".format(matt))


print(" ")
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
print("Training data loaded.")
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

if True:
	print(" ")
	#print(len(Xtest))

	print("====================================")
	print("Confidence Only: (Classification)")
	print("====================================")

	print(" ")

	
	print("-------------------------------------")
	print("Neural Network: (Threshold 0.5)")
	print(" ")
	print(" ")
	clf = load_model("paper_models/conf_baseline/conf_baseline.model")
	print(" ")
	print(" ")
	class_evaluate_model_threshold()
	
	
	print("-------------------------------------")

	print("Support Vector Machine with RBF Kernel:")
	if(os.path.isfile("SVM_RBF.sav") ):
		clf = pickle.load(open("SVM_RBF.sav", 'rb'))
	else:
		clf = svm.SVC(verbose=True)
		clf.fit(Xtrain,Ytrain)
		pickle.dump(clf, open("SVM_RBF.sav", 'wb'))
	class_evaluate_model()

	print("-------------------------------------")

	print("Support Vector Machine with Sigmoid Kernel:")
	if(os.path.isfile("SVM_Sigmoid.sav") ):
		clf = pickle.load(open("SVM_Sigmoid.sav", 'rb'))
	else:
		clf = svm.SVC(kernel='sigmoid',verbose=True)
		clf.fit(Xtrain,Ytrain)
		pickle.dump(clf, open("SVM_Sigmoid.sav", 'wb'))
	class_evaluate_model()


	print("-------------------------------------")

	print("Support Vector Machine with Linear Kernel and L2 Penalization:")
	if(os.path.isfile("SVM_Linear_L2.sav") ):
		clf = pickle.load(open("SVM_Linear_L2.sav", 'rb'))
	else:
		clf = svm.LinearSVC(verbose=True)
		clf.fit(Xtrain,Ytrain)
		pickle.dump(clf, open("SVM_Linear_L2.sav", 'wb'))
	class_evaluate_model()

	print("-------------------------------------")
	print("Vanilla Logistic Regression")
	clf = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6,verbose=1,n_jobs=4,solver='lbfgs')
	clf.fit(Xtrain, Ytrain)
	class_evaluate_model()

	print("-------------------------------------")
	print("Vanilla Bayesian Ridge Regression (Threshold 0.5)")
	clf = linear_model.BayesianRidge(verbose=True)
	clf.fit(Xtrain, Ytrain)
	class_evaluate_model_threshold()

	print("-------------------------------------")
	print("Gaussian Naive Bayes")
	clf = GaussianNB()
	clf.fit(Xtrain, Ytrain)
	class_evaluate_model()

	print("-------------------------------------")
	print("Gaussian Process Classifier (trained on 10k samples)")
	if(os.path.isfile("GPC.sav") ):
		clf = pickle.load(open("GPC.sav", 'rb'))
	else:
		clf = GaussianProcessClassifier()
		clf.fit(Xtrain[0:9999,:], Ytrain[0:9999])
		pickle.dump(clf, open("GPC.sav", 'wb'))
	class_evaluate_model()


	print("-------------------------------------")
	print("Vanilla Decision Tree")
	clf = tree.DecisionTreeClassifier()
	clf.fit(Xtrain, Ytrain)
	class_evaluate_model()

	print("-------------------------------------")
	print("Vanilla Random Forest")
	clf = RandomForestClassifier(n_jobs=4)
	clf.fit(Xtrain, Ytrain)
	class_evaluate_model()

	print("-------------------------------------")
	print("AdaBoosted Decision Free")
	clf = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=1),
	                         algorithm="SAMME",
	                         n_estimators=200)
	clf.fit(Xtrain, Ytrain)
	class_evaluate_model()

	print("-------------------------------------")
	print("XGboost")
	clf = xgb.XGBClassifier(silent=False, n_jobs=4, n_estimators=100, max_depth=1, learning_rate=0.1, subsample=0.5)
	clf.fit(Xtrain, Ytrain)
	class_evaluate_model_threshold()

	params = {
    'objective' :'binary',
    'learning_rate' : 0.02,
    'num_leaves' : 76,
    'feature_fraction': 0.64, 
    'bagging_fraction': 0.8, 
    'bagging_freq':1,
    'boosting_type' : 'gbdt',
    'metric': 'binary_logloss'
	}

	print("-------------------------------------")
	print("LightGBM (Threshold 0.5)")
	Xt, Xv, Yt, Yv = train_test_split(Xtrain,Ytrain,test_size=0.2)
	d_train = lgbm.Dataset(Xt, Yt)
	d_valid = lgbm.Dataset(Xv, Yv)
	clf = lgbm.train(params, d_train, 5000, valid_sets=[d_valid], verbose_eval=50, early_stopping_rounds=100)
	class_evaluate_model_threshold()

	

if True:
	print("====================================")
	print("Estimation Only: (Regression)")
	print("====================================")

	print(" ")
	print("Reading training data...")
	Xtrain=pd.read_csv('data/jacob_feature_train.csv' ,dtype='double')
	Xtrain=scale(Xtrain.dropna(axis=1).loc[:, ~(Xtrain == 0).any(0)])

	print(" ")
	print("Sanity Check:")
	print(" ")
	print(Xtrain)
	Ytrain=pd.read_csv('data/jacob_label_train.csv' ,dtype='double').dropna(axis=1).values
	Ytrain=minmax_scale(Ytrain[:, ~(Ytrain == 0).any(0)],feature_range=(-1,1))
	print(" ")
	print(Ytrain)
	print("Training data loaded.")
	print(" ")

	print("Reading test data...")
	Xtest=pd.read_csv('data/jacob_feature_test.csv' ,dtype='double')
	Xtest=scale(Xtest.dropna(axis=1).loc[:, ~(Xtest == 0).any(0)])

	Ytest=pd.read_csv('data/jacob_label_test.csv' ,dtype='double').dropna(axis=1).values
	Ytest=minmax_scale(Ytest[:, ~(Ytest == 0).any(0)],feature_range=(-1,1))
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

	print("-------------------------------------")
	print("Neural Network: ")
	print(" ")
	print(" ")
	clf = nn_1(len(Xtrain[0,:]))
	clf.load_weights("jacob_baseline.hdf5")
	#clf = load_model("jacob_baseline.hdf5")
	print(" ")
	print(" ")
	reg_evaluate_model()

	print("-------------------------------------")
	print("Linear Regression")
	clf = linear_model.LinearRegression(n_jobs=4)
	clf.fit(Xtrain, Ytrain)
	reg_evaluate_model()

	print("-------------------------------------")
	print("Ridge Regression")
	clf = linear_model.Ridge()
	clf.fit(Xtrain, Ytrain)
	reg_evaluate_model()

	
	print("-------------------------------------")
	print("Vanilla Decision Tree")
	clf = tree.DecisionTreeRegressor()
	clf.fit(Xtrain, Ytrain)
	reg_evaluate_model()
	

	print("-------------------------------------")
	print("Gaussian Process Regression (trained on 10k samples)")
	if(os.path.isfile("GPR.sav") ):
		clf = pickle.load(open("GPR.sav", 'rb'))
	else:
		clf = GaussianProcessRegressor()
		clf.fit(Xtrain[0:9999,:], Ytrain[0:9999,:])
		pickle.dump(clf, open("GPR.sav", 'wb'))
	reg_evaluate_model()

	
	print("-------------------------------------")
	print("Nearest Neighbors Regression")
	clf = KNeighborsRegressor()
	clf.fit(Xtrain, Ytrain)
	reg_evaluate_model()
	
