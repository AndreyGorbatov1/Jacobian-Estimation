from keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize, scale
Xtest=pd.read_csv('data/conf_feature_test.csv' ,dtype='double')
size = len(Xtest)
Xtest=Xtest.dropna(axis=1)
p560=[0,0,0.15,0.4318,0,0 , 0 ,0.4318 ,0.0203 ,0 ,0 ,0   ,1.5708 ,0 ,-1.5708, 1.5708, -1.5708, 0]
cartes = []
"""
for x in np.arange(-0.4,0.4,0.05):
	for y in np.arange(-0.4,0.4,0.05):
		for z in np.arange(-0.4,0.4,0.05):
			#print(p560+[x,y,z,0,0,0])
			cartes.append(np.array([x,y,z]))
			df=pd.DataFrame([p560+[x,y,z,0.1,0.1,0.1]],columns=['d1','d2','d3','d4','d5','d6','a1','a2','a3','a4','a5','a6','alpha1','alpha2','alpha3','alpha4','alpha5','alpha6','x','y','z','wx','wy','wz'])
			Xtest=pd.concat([Xtest,df])
"""
for wx in np.arange(np.pi,-np.pi,0.25):
	for wy in np.arange(np.pi,-np.pi,0.25):
		for wz in np.arange(np.pi,-np.pi,0.25):
			#print(p560+[x,y,z,0,0,0])
			cartes.append(np.array([wx,wy,wz]))
			df=pd.DataFrame([p560+[0.5,0.5,0.5,wx,wy,wz]],columns=['d1','d2','d3','d4','d5','d6','a1','a2','a3','a4','a5','a6','alpha1','alpha2','alpha3','alpha4','alpha5','alpha6','x','y','z','wx','wy','wz'])
			Xtest=pd.concat([Xtest,df])
print(Xtest)
Xtest=scale(Xtest)
Xtest=np.hstack((Xtest[size:len(Xtest),2:4],Xtest[size:len(Xtest),7:9],Xtest[size:len(Xtest),12:13],Xtest[size:len(Xtest),14:17],Xtest[size:len(Xtest),18:24]))
cartes=np.array(cartes)
model = load_model('conf_matlab.hdf5')
res = model.predict(Xtest)
print(res)
np.savetxt('p560_ow.csv',res,delimiter=',')
np.savetxt('p560_ow_cartes.csv',cartes,delimiter=',')