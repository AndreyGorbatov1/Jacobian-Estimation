import pickle
import pandas as pd
import numpy as np
import os

from matplotlib import pyplot as plt
from matplotlib import colors as colors
import seaborn as sns

def plot(file_name):
    raw = pd.read_pickle("conf_comparison/"+file_name)
    optimizer = file_name[0:file_name.index("_")+1]
    optimizer = optimizer[file_name.index("_"): len(optimizer)]
    raw['epoch'] = list(range(0, len(raw['loss'])))
    raw = pd.DataFrame.from_dict(raw)
    print(raw)

    loss = raw.drop("val_loss", 1)
    valloss = raw.drop("loss", 1)

    loss['loss'] = np.log(loss['loss'])
    valloss['val_loss'] = np.log(valloss['val_loss'])
    
    loss = sns.lineplot(x="epoch", y="loss", label=optimizer+"_train", data=loss)
    val_loss = sns.lineplot(x="epoch", y="val_loss", label=optimizer+"_validation", data=valloss)
    loss.set(xlabel='Epoch', ylabel='Loss (log scale)')
    val_loss.set(xlabel='Epoch', ylabel='Loss (log scale)')

    return loss, val_loss

def plot_total():
    loss_total = []
    valloss_total = []
    for filename in os.listdir("conf_comparison"):
        if "model" not in filename:
            loss, valloss = plot(filename)
            loss_total.append(loss)
            valloss_total.append(valloss)
    
    for subplot in loss_total:
        plt.legend()
        plt.subplot(subplot)
    plt.show()
    
if __name__ == '__main__':
    plot_total()
