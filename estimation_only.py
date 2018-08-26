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
import pickle
from sklearn.metrics import r2_score
#a=pd.read_csv('featureTrain.csv' ,dtype='double')
#print(a)

import tensorflow as tf
def huber_loss(y_true, y_pred):
    return tf.losses.huber_loss(y_true,y_pred)

sys.path.append('/home/alexanderliao/data/GitHub/')
from kerasresnet import resnet


def nn_1(input_length):
    model = Sequential()
    model = Sequential()
    model.add(Dense(32, input_dim=input_length, kernel_initializer='RandomUniform'))
    ##model.add(BatchNormalization())
    model.add(PReLU())
    #model.add(Dropout(0.25))

    model.add(Dense(64, input_dim=32, kernel_initializer='RandomUniform'))
    ##model.add(BatchNormalization())
    model.add(PReLU())
    #model.add(Dropout(0.25))

    model.add(Dense(128, input_dim=64, kernel_initializer='RandomUniform'))
    ##model.add(BatchNormalization())
    model.add(PReLU())
    #model.add(Dropout(0.25))

    model.add(Dense(256, input_dim=128, kernel_initializer='RandomUniform'))
    #model.add(BatchNormalization())
    model.add(PReLU())
    #model.add(Dropout(0.25))

    
    model.add(Dense(512, input_dim=256, kernel_initializer='RandomUniform'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.5))


    model.add(Dense(1050, input_dim=512, kernel_initializer='RandomUniform'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.5))

    
    model.add(Dense(2150, input_dim=1050, kernel_initializer='RandomUniform'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.5))

    model.add(Dense(1050, input_dim=2150, kernel_initializer='RandomUniform'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.5))
    
    model.add(Dense(512, input_dim=1050, kernel_initializer='RandomUniform'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.5))
    
    model.add(Dense(512, input_dim=1024, kernel_initializer='RandomUniform'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.5))

    model.add(Dense(512, input_dim=512, kernel_initializer='RandomUniform'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.5))

    model.add(Dense(256, input_dim=512, kernel_initializer='RandomUniform'))
    #model.add(BatchNormalization())
    model.add(PReLU())
    #model.add(Dropout(0.5))


    model.add(Dense(128, input_dim=256, kernel_initializer='RandomUniform'))
    ##model.add(BatchNormalization())
    model.add(PReLU())
    #model.add(Dropout(0.5))

    model.add(Dense(64, input_dim=128, kernel_initializer='RandomUniform'))
    ##model.add(BatchNormalization())
    model.add(PReLU())
    #model.add(Dropout(0.5))

    model.add(Dense(32, input_dim=64, kernel_initializer='RandomUniform'))
    ##model.add(BatchNormalization())
    model.add(Dense(25, activation="linear"))

    #Dense(64, input_dim=24, kernel_initializer="RandomUniform")`    
    opt = optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0001, nesterov=False)

    model.compile(optimizer="adam", loss="mean_squared_error")
    #model.compile(optimizer="adam", loss="softmax")

    return model


def routine(Ytest,nn_predictor):
    acc=r2_score(Ytest,nn_predictor.predict(Xtest))
    print(acc)
    string=str(datetime.now()).replace(".","").replace(" ","")+'-'+str(round(acc,2))
    nn_predictor.save(string+'.model')
    shutil.move(string+'.model','./models/'+string+'.model')
    #print(r2_score(Ytest,nn_predictor.predict(Xtest)))
    return string

def baseline(Ytest,nn_predictor):
    acc=r2_score(Ytest,nn_predictor.predict(Xtest))
    print(acc)
    string="jacob_baseline"
    nn_predictor.save(string+'.model')
    #shutil.move(string+'.model','./models/'+string+'.model')
    #print(r2_score(Ytest,nn_predictor.predict(Xtest)))
    return string

if __name__ == "__main__":

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
	#print("Seed: {}".format(numpy.random.seed(7)))
	#Xtrain=numpy.nonzero(numpy.loadtxt('featureTrain.csv',dtype='float32',delimiter=','))
	#Ytrain=numpy.loadtxt('labelTrain.csv',dtype='float32',delimiter=',')
	#Xtrain=Xtrain[1:30000,:]
	#Ytrain=Ytrain[1:30000,:]
	#Ｙtrain=minmax_scale(Ytrain, feature_range=(0, 1), axis=0, copy=True)
	#Xtrain=normalize(Xtrain,axis=1)
	#Xtrain = scale( Xtrain, axis=0, with_mean=True, with_std=True, copy=True )
	#Ytrain = scale( Ytrain, axis=0, with_mean=True, with_std=True, copy=True )
	#print(type(Xtrain))
	#print(Ytrain.shape)
	#Xtest=numpy.nonzero(numpy.loadtxt('featureTest.csv',dtype='float32',delimiter=','))
	#Ytest=numpy.loadtxt('labelTest.csv',dtype='float32',delimiter=',')
	#Xtest=Xtest[1:30000,:]
	#Ytest=Ytest[1:30000,:]
	#Ｙtest=minmax_scale(Ytest, feature_range=(0, 1), axis=0, copy=True)
	#Xtest=normalize(Xtest,axis=1)
	#Xtest = scale( Xtest, axis=0, with_mean=True, with_std=True, copy=True )
	#Ytest = scale( Ytest, axis=0, with_mean=True, with_std=True, copy=True )
	#print(type(Xtrain))
	#print(type(Ytrain))

	#nn_predictor=resnet.ResnetBuilder().build_resnet_18([1,1,14],1)
	#nn_predictor=resnet.ResnetBuilder().build([1,1,14],1,'basic_block',8)
	#print(nn_predictor.input_shape)
	#print(nn_predictor.output_shape)

	opt = optimizers.RMSprop(lr=0.01)

	#nn_predictor.compile(optimizer="adadelta", loss="binary_crossentropy")

	nn_predictor = nn_1(len(Xtrain[0,:]))
	print(len(Xtrain[0,:]))
	print(nn_predictor.summary())

	#print("Cleaning directories...")
	#os.system("rm -r graph")
	#os.system("mkdir graph")

	b_size = 4096
	epoch = 750
	val_split = 0.2

	#early_stopping = keras.callbacks.EarlyStopping(patience=50, verbose=1)
	model_checkpoint = keras.callbacks.ModelCheckpoint("./est_baseline.model", save_best_only=True, verbose=1)
	#reduce_lr = keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1)
	tensorboard=keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
	callbacks = [model_checkpoint,tensorboard]

	with tf.device('/gpu:0'):
	    try:
	        callback=keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
	        print(Ytrain.shape)
	        print(Xtrain.shape)
	        print("Batch Size: {}".format(b_size) )
	        print("Epochs: {}".format(epoch) )
	        print("Validation Split: {}".format(val_split) )

	        history=nn_predictor.fit(Xtrain,Ytrain, batch_size=b_size, epochs=epoch, validation_split=val_split,verbose=1, callbacks=callbacks, shuffle=True)
	        nn_predictor.save("est_baseline.model")
	        with open('./est_hist', 'wb') as file_pi:
                    pickle.dump(history.history, file_pi)
	        print(type(history))
	        #baseline(Ytest,nn_predictor)
	        string=routine(Ytest,nn_predictor)
	        json.dump(history.history, open( string+".json", "w" ))
	        shutil.move(string+'.json','./histories/'+string+'.pickle')
	    except (KeyboardInterrupt, SystemExit):
	        routine(Ytest,nn_predictor)
	        #baseline(Ytest,nn_predictor)



	#print(nn_predictor.predict([0.15,0.4318,0.4318,0.0203,1.5708,-1.5708,1.5708,-1.5708,0,17.4,4.8,0.82,0.34,0.09,0.3638,0.006,0.2275,-0.0203,-0.0141,0.07,0,0.019,0,0,0,0,0,0,0.032]))