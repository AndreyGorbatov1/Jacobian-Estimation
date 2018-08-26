# Visualize training history
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy
import pickle
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
with open('paper_models/est_baseline/est_hist', 'rb') as fid:
    hist = pickle.load(fid)
# list all data in history
print(hist.keys())
# summarize history for accuracy

plt.xlabel('epoch')

# summarize history for loss
plt.plot(hist['loss'])
plt.plot(hist['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()
