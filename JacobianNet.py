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
from keras.models import load_model
import tensorflow as tf

def huber_loss(y_true, y_pred):
    return tf.losses.huber_loss(y_true,y_pred)

def compute_r2(Ytest,Xtest,nn_predictor):
    acc=r2_score(Ytest,nn_predictor.predict(Xtest))
    print(acc)
    string=str(datetime.now()).replace(".","").replace(" ","")+'-'+str(round(acc,2))
    #nn_predictor.save(string+'.model')
    #shutil.move(string+'.model','./models/'+string+'.model')
    #print(r2_score(Ytest,nn_predictor.predict(Xtest)))
    return string

sys.path.append('/home/alexanderliao/data/GitHub/')


print("Reading confidence training data...")
conf_Xtrain=pd.read_csv('data/conf_feature_dyna_train.csv' ,dtype='double')
conf_Xtrain=scale(conf_Xtrain.dropna(axis=1).loc[:, ~(conf_Xtrain == 0).any(0)])
conf_Ytrain=pd.read_csv('data/conf_label_dyna_train.csv' ,dtype='double').dropna(axis=1)
print("Done!")

print("Reading confidence test data...")
conf_Xtest=pd.read_csv('data/conf_feature_dyna_test.csv' ,dtype='double')
conf_Xtest=scale(conf_Xtest.dropna(axis=1).loc[:, ~(conf_Xtest == 0).any(0)])

conf_Ytest=pd.read_csv('data/conf_label_dyna_test.csv' ,dtype='double').dropna(axis=1)
print("Done!")

print("Reading estimation training data...")
jacob_Xtrain=pd.read_csv('data/jacob_feature_train.csv' ,dtype='double')
jacob_Xtrain=scale(jacob_Xtrain.dropna(axis=1).loc[:, ~(jacob_Xtrain == 0).any(0)])
print(" ")

jacob_Ytrain=pd.read_csv('data/jacob_label_train.csv' ,dtype='double').dropna(axis=1)
jacob_Ytrain=minmax_scale(jacob_Ytrain.loc[:, ~(jacob_Ytrain == 0).any(0)],feature_range=(-1,1))
print("Done!")

print("Reading estimation test data...")
jacob_Xtest=pd.read_csv('data/jacob_feature_test.csv' ,dtype='double')
jacob_Xtest=scale(jacob_Xtest.dropna(axis=1).loc[:, ~(jacob_Xtest == 0).any(0)])

jacob_Ytest=pd.read_csv('data/jacob_label_test.csv' ,dtype='double').dropna(axis=1)
jacob_Ytest=minmax_scale(jacob_Ytest.loc[:, ~(jacob_Ytest == 0).any(0)],feature_range=(-1,1))
print("Done!")

numpy.random.seed(7)


def encoder(feed):
    inp=Input(shape=(len(feed[0,:]),) )

    f=Dense(32, kernel_initializer='RandomUniform')(inp)
    ##f=BatchNormalization()(f)
    f=PReLU()(f)
    #f=Dropout(0.25)(f)

    f=Dense(64, input_dim=32, kernel_initializer='RandomUniform')(f)
    ##f=BatchNormalization()(f)
    f=PReLU()(f)
    #f=Dropout(0.25)(f)

    f=Dense(128, input_dim=64, kernel_initializer='RandomUniform')(f)
    ##f=BatchNormalization()(f)
    f=PReLU()(f)
    #f=Dropout(0.25)(f)

    f=Dense(256, input_dim=128, kernel_initializer='RandomUniform')(f)
    #f=BatchNormalization()(f)
    f=PReLU()(f)
    #f=Dropout(0.25)(f)
   
    f=Dense(512, input_dim=256, kernel_initializer='RandomUniform')(f)
    f=BatchNormalization()(f)
    f=PReLU()(f)
    f=Dropout(0.5)(f)

    f=Dense(1050, input_dim=512, kernel_initializer='RandomUniform')(f)
    f=BatchNormalization()(f)
    f=PReLU()(f)
    f=Dropout(0.5)(f)
    
    f=Dense(2150, input_dim=1050, kernel_initializer='RandomUniform')(f)
    f=BatchNormalization()(f)
    f=PReLU()(f)
    f=Dropout(0.5)(f)
    return inp, f

def conf_net(encoded):
    f=Dense(1050, input_dim=2150, kernel_initializer='RandomUniform')(encoded.output)
    f=BatchNormalization()(f)
    f=PReLU()(f)
    f=Dropout(0.5)(f)
    
    f=Dense(512, input_dim=1050, kernel_initializer='RandomUniform')(f)
    f=BatchNormalization()(f)
    f=PReLU()(f)
    f=Dropout(0.5)(f)
    
    f=Dense(512, input_dim=1024, kernel_initializer='RandomUniform')(f)
    f=BatchNormalization()(f)
    f=PReLU()(f)
    f=Dropout(0.5)(f)

    f=Dense(512, input_dim=512, kernel_initializer='RandomUniform')(f)
    f=BatchNormalization()(f)
    f=PReLU()(f)
    f=Dropout(0.5)(f)

    f=Dense(256, input_dim=512, kernel_initializer='RandomUniform')(f)
    #f=BatchNormalization()(f)
    f=PReLU()(f)
    #f=Dropout(0.5)(f)


    f=Dense(128, input_dim=256, kernel_initializer='RandomUniform')(f)
    ##f=BatchNormalization()(f)
    f=PReLU()(f)
    #f=Dropout(0.5)(f)

    f=Dense(64, input_dim=128, kernel_initializer='RandomUniform')(f)
    ##f=BatchNormalization()(f)
    f=PReLU()(f)
    #f=Dropout(0.5)(f)

    f=Dense(32, input_dim=64, kernel_initializer='RandomUniform')(f)
    ##f=BatchNormalization()(f)
    f=Dense(1, activation="sigmoid")(f)
    return f

def est_net(encoded):
    f=Dense(1050, input_dim=2150, kernel_initializer='RandomUniform')(encoded)
    f=BatchNormalization()(f)
    f=PReLU()(f)
    f=Dropout(0.5)(f)
    
    f=Dense(512, input_dim=1050, kernel_initializer='RandomUniform')(f)
    f=BatchNormalization()(f)
    f=PReLU()(f)
    f=Dropout(0.5)(f)
    
    f=Dense(512, input_dim=1024, kernel_initializer='RandomUniform')(f)
    f=BatchNormalization()(f)
    f=PReLU()(f)
    f=Dropout(0.5)(f)

    f=Dense(512, input_dim=512, kernel_initializer='RandomUniform')(f)
    f=BatchNormalization()(f)
    f=PReLU()(f)
    f=Dropout(0.5)(f)

    f=Dense(256, input_dim=512, kernel_initializer='RandomUniform')(f)
    #f=BatchNormalization()(f)
    f=PReLU()(f)
    #f=Dropout(0.5)(f)

    f=Dense(128, input_dim=256, kernel_initializer='RandomUniform')(f)
    ##f=BatchNormalization()(f)
    f=PReLU()(f)
    #f=Dropout(0.5)(f)

    f=Dense(64, input_dim=128, kernel_initializer='RandomUniform')(f)
    ##f=BatchNormalization()(f)
    f=PReLU()(f)
    #f=Dropout(0.5)(f)

    f=Dense(32, input_dim=64, kernel_initializer='RandomUniform')(f)
    ##f=BatchNormalization()(f)
    f=Dense(25, activation="linear")(f)
    return f

inp,encoded = encoder(conf_Xtrain)
encoding = Model(inp, encoded)
estimation = est_net(encoded)
estimation_model = Model(inp, estimation)
estimation_model.compile(optimizer="adam", loss="mean_squared_error")

#print(estimation_model.layers[1].trainable)
#print(confidence_model.layers[1].trainable)
#print(confidence_model.layers)
#print(type(confidence_model.layers[1]))
#print(isinstance(confidence_model.layers[1],keras.layers.core.Dense))

print("---------------------------------------------")
print("Cleaning directories...")
os.system("rm -r graph")
os.system("mkdir graph")

b_size = 4096
epoch = 25
val_split = 0.2



early_stopping = keras.callbacks.EarlyStopping(patience=50, verbose=1)
model_checkpoint = keras.callbacks.ModelCheckpoint("./intm.model", save_best_only=True, verbose=1)
reduce_lr = keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=10, min_lr=0.00001, verbose=1)
tensorboard=keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
callbacks = [early_stopping,model_checkpoint,tensorboard]


model_checkpoint2 = keras.callbacks.ModelCheckpoint("./pretrained_checkpt.model", save_best_only=True, verbose=1)
callbacks2 = [early_stopping,model_checkpoint2,tensorboard]

print("==============================================")
print("Start training: check progress by typing: tensorboard --logdir ./graph --port 6028")
print("---------------------------------------------")
print("Pre-training encoder on estimation net:")
with tf.device('/gpu:0'):
	try:
		if(os.path.isfile("pretrained.model") ):
			print("Pretrained model found, loading...")
			print("  ")
			encoding=load_model("pretrained.model")
			print("  ")
		else:
			print("Pretrained model not found, training...")
			pretrained_hist=estimation_model.fit(jacob_Xtrain,jacob_Ytrain, batch_size=b_size, epochs=500, validation_split=val_split,verbose=1, callbacks=callbacks2, shuffle=True)
			encoding.save("pretrained.model")
			json.dump(pretrained_hist, open("./histories/pretrained_hist.json", 'w'))
		print("Dataset shape: (X;Y)")
		print(jacob_Xtrain.shape)
		print(jacob_Ytrain.shape)
		print("Batch Size: {}".format(b_size) )
		print("Epochs: {}".format(epoch) )
		print("Validation Split: {}".format(val_split) )

		#print(type(history))
		#baseline(Ytest,nn_predictor)
		print("Estimation net R2:")
		compute_r2(jacob_Ytest,jacob_Xtest,estimation_model)
		#print("Confidence net R2:")
		#compute_r2(conf_Ytest,conf_Xtest,confidence_model)
		#json.dump(history.history, open( string+".json", "w" ))
		#shutil.move(string+'.json','./histories/'+string+'.pickle')
	except (KeyboardInterrupt, SystemExit):
		compute_r2(jacob_Ytest,jacob_Xtest,estimation_model)
		#compute_r2(conf_Ytest,conf_Xtest,confidence_model)
		#baseline(Ytest,nn_predictor)

print("Freezing encoder and compiling confidence net...")
encoding.trainable=False
confidence = conf_net(encoding)
confidence_model= Model(inp,confidence)
confidence_model.compile(optimizer="adam", loss="binary_crossentropy")

print("==============================================")
print("Training confidence net and fine-tuning estimation net:")
for l in range(10):

	print("---------------------------------------------")
	print("Training confidence net:")
	i = 0 
	

	with tf.device('/gpu:0'):
		try:
			print("Dataset shape: (X;Y)")
			print(conf_Xtrain.shape)
			print(conf_Ytrain.shape)
			print("Batch Size: {}".format(b_size) )
			print("Epochs: {}".format(epoch) )
			print("Validation Split: {}".format(val_split) )

			conf_history=confidence_model.fit(conf_Xtrain,conf_Ytrain, batch_size=b_size, epochs=epoch, validation_split=val_split,verbose=1, callbacks=callbacks, shuffle=True)

			print(type(history))
			json.dump(conf_history, open("./histories/conf_hist"+str(l)+".json", 'w'))

			print("Estimation net R2:")
			compute_r2(jacob_Ytest,jacob_Xtest,estimation_model)
			print("Confidence net R2:")
			compute_r2(conf_Ytest,conf_Xtest,confidence_model)
			#json.dump(history.history, open( string+".json", "w" ))
			#shutil.move(string+'.json','./histories/'+string+'.pickle')
		except (KeyboardInterrupt, SystemExit):
			compute_r2(conf_Ytest,conf_Xtest,confidence_model)


	print("---------------------------------------------")
	print("Training estimation net:")
	i=0
	#print("Unfreezing encoder...")
		#if isinstance(layer, keras.layers.core.Dense) and i < 21:
		#	layer.trainable = True
		#i = i+1
	#for layer in confidence_model.layers:
		#if i < 21: layer.trainable = False
		#else: break
		#i=i+1

	with tf.device('/gpu:0'):
		try:
			print("Dataset shape: (X;Y)")
			print(jacob_Xtrain.shape)
			print(jacob_Ytrain.shape)
			print("Batch Size: {}".format(b_size) )
			print("Epochs: {}".format(epoch) )
			print("Validation Split: {}".format(val_split) )

			est_history=estimation_model.fit(jacob_Xtrain,jacob_Ytrain, batch_size=b_size, epochs=epoch, validation_split=val_split,verbose=1, callbacks=callbacks2, shuffle=True)
			json.dump(est_history, open("./histories/est_hist"+str(l)+".json", 'w'))
			#baseline(Ytest,nn_predictor)

			print("Estimation net R2:")
			compute_r2(jacob_Ytest,jacob_Xtest,estimation_model)
			print("Confidence net R2:")
			compute_r2(conf_Ytest,conf_Xtest,confidence_model)
			#json.dump(history.history, open( string+".json", "w" ))
			#shutil.move(string+'.json','./histories/'+string+'.pickle')
		except (KeyboardInterrupt, SystemExit):
			compute_r2(jacob_Ytest,jacob_Xtest,estimation_model)
			compute_r2(conf_Ytest,conf_Xtest,confidence_model)
			#baseline(Ytest,nn_predictor)

