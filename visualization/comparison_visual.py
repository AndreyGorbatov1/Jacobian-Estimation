import pickle
import pandas as pd
import numpy as np
import os

from matplotlib import pyplot as plt
from matplotlib import colors as colors
import seaborn as sns

def plot_loss(file_name, count):
    raw = pd.read_pickle("est_comparison/"+file_name)
    optimizer = file_name[file_name.index("_")+1:len(file_name)]
    optimizer = optimizer[0:optimizer.index("_")]
    raw['epoch'] = list(range(0, len(raw['loss'])))
    raw = pd.DataFrame.from_dict(raw)
    print(raw)

    loss = raw.drop("val_loss", 1)
    print(loss)

    loss['loss'].rolling(5)
    loss['loss'] = np.log(loss['loss'])

    with sns.axes_style("darkgrid"):
        sns.set_palette(sns.color_palette("hls",8))
        loss = sns.lineplot(x="epoch", y="loss", label=optimizer, data=loss, dashes=True)
        if not count%2 == 0:
            loss.lines[count].set_linestyle("--")        
        loss.set(xlabel='Epoch', ylabel='Loss (log scale)')

    return loss

def plot_valloss(file_name):
    raw = pd.read_pickle("conf_comparison/"+file_name)
    optimizer = file_name[0:file_name.index("_")+1]
    optimizer = optimizer[file_name.index("_"): len(optimizer)]
    raw['epoch'] = list(range(0, len(raw['loss'])))
    raw = pd.DataFrame.from_dict(raw)
    print(raw)

    valloss = raw.drop("loss", 1)

    valloss['val_loss'] = np.log(valloss['val_loss'])

    val_loss = sns.lineplot(x="epoch", y="val_loss", label="validation", data=valloss)
    val_loss.set(xlabel='Epoch', ylabel='Loss (log scale)')

    return val_loss
    
def plot_total():
    count = 0
    loss_total = []
    valloss_total = []
    optimizers = []
    for filename in os.listdir("est_comparison"):
        if "model" not in filename:
            loss = plot_loss(filename, count)
            loss_total.append(loss)
            count = count+1
            #valloss = plot_valloss(filename)
            #valloss_total.append(valloss)
    
    print("Size of loss_total: " + str(len(loss_total)))
    print("Size of valloss_total: " + str(len(valloss_total)))

    for loss_plt in loss_total:
        plt.subplot(loss_plt)
    plt.show()
    
if __name__ == '__main__':
    plot_total()
