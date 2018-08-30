import pandas as pd
import numpy as np
from tensorboard.backend.event_processing import event_accumulator as ea

from matplotlib import pyplot as plt
from matplotlib import colors as colors
import seaborn as sns
sns.set(style="darkgrid")
sns.set_context("paper")

def plot():
  loss_raw_data = pd.read_csv("Conf_Base/run_.-tag-loss.csv")
  loss_raw_data.drop("Wall time", 1, inplace=True)
  loss_raw_data['Value'] = np.log(loss_raw_data['Value'])
  print(loss_raw_data)
  valloss_raw_data = pd.read_csv("Conf_Base/run_.-tag-val_loss.csv")
  valloss_raw_data.drop("Wall time", 1, inplace=True)
  valloss_raw_data['Value'] = np.log(valloss_raw_data['Value'])
  print(valloss_raw_data)
  
  loss = sns.lineplot(x="Step", y="Value", label="Train", data=loss_raw_data)
  val_loss = sns.lineplot(x="Step", y="Value", label="Validation", data=valloss_raw_data)
  loss.set(xlabel='Epoch', ylabel='Loss (log scale)')
  val_loss.set(xlabel='Epoch', ylabel='Loss (log scale)')

  plt.legend()
  plt.subplot(loss)
  plt.subplot(val_loss)
  plt.show()


if __name__ == '__main__':
  plot()
