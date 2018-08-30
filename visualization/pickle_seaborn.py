import pickle
import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib import colors as colors


#load data
raw_conf = []
for i in range(0, 11):
    temp = pd.read_pickle("conf_history/conf_history"+str(i))
    temp['epoch'] = list(range(0+i*10, len(temp['loss'])+i*10))
    temp = pd.DataFrame.from_dict(temp)
    raw_conf.append(temp)
raw_conf = pd.concat(raw_conf)

raw_est = []
for i in range(0, 10):
    temp = pd.read_pickle("est_history/est_history"+str(i))
    temp['epoch'] = list(range(0+i*5, len(temp['loss'])+i*5))
    temp = pd.DataFrame.from_dict(temp)
    raw_est.append(temp)
raw_est = pd.concat(raw_est)

#classify
loss_est = raw_est.drop("val_loss", 1)
valloss_est = raw_est.drop("loss", 1)

loss_conf = raw_conf.drop("val_loss", 1)
valloss_conf = raw_conf.drop("loss", 1)

#log
loss_conf['loss'] = np.log(loss_conf['loss'])
valloss_conf['val_loss'] = np.log(valloss_conf['val_loss'])

loss_est['loss'] = np.log(loss_est['loss'])
valloss_est['val_loss'] = np.log(valloss_est['val_loss'])

def plot_conf():
    loss_conf_plt = sns.lineplot(x='epoch', y='loss', label='loss', data=loss_conf)
    valloss_conf_plt = sns.lineplot(x='epoch', y='val_loss', label='val_loss', data=valloss_conf)

    loss_conf_plt.set(xlabel='Epoch', ylabel='Loss (log scale)')
    valloss_conf_plt.set(xlabel='Epoch', ylabel='Loss (log scale)')
    return loss_conf_plt,valloss_conf_plt

def plot_est():
    loss_est_plt = sns.lineplot(x='epoch', y='loss', label='loss', data=loss_est)
    valloss_est_plt = sns.lineplot(x='epoch', y='val_loss', label='val_loss', data=valloss_est)

    loss_est_plt.set(xlabel='Epoch', ylabel='Loss (log scale)')
    valloss_est_plt.set(xlabel='Epoch', ylabel='Loss (log scale)')
    return loss_est_plt, valloss_est_plt

def plot():
    plt.legend()
    plot = plot_conf()
    plt.subplot(plot[0])
    plt.subplot(plot[1])
    plt.show()

if __name__ == '__main__':
    plot()