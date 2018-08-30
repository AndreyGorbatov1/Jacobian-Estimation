import pickle
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import colors as colors
import seaborn as sns

def plot():
    raw = pd.read_pickle("Est_Base/est_hist")
    raw['epoch'] = list(range(0, len(raw['loss'])))
    raw = pd.DataFrame.from_dict(raw)
    print(raw)

    loss = raw.drop("val_loss", 1)
    valloss = raw.drop("loss", 1)

    loss['loss'] = np.log(loss['loss'])
    valloss['val_loss'] = np.log(valloss['val_loss'])
    
    loss = sns.lineplot(x="epoch", y="loss", label="Train", data=loss)
    val_loss = sns.lineplot(x="epoch", y="val_loss", label="Validation", data=valloss)
    loss.set(xlabel='Epoch', ylabel='Loss (log scale)')
    val_loss.set(xlabel='Epoch', ylabel='Loss (log scale)')

    plt.subplot(loss)
    plt.subplot(val_loss)
    plt.show()


if __name__ == '__main__':
  plot()
