# coding: utf-8

"""
Plot accuracy and loss for a trained model

@Author: Tony Di Pilato

Created on Fri Dec 13, 2019
"""


import os
import numpy as np
from libtiff import TIFF
import pandas as pd
import matplotlib.pyplot as plt

plot_dir = '../plots/'
save_dir = '../models/'
model_name = 'EF-UNet_256-64_sklw-bce'
history_name = model_name + '_history'

history = pd.read_hdf(save_dir + history_name + ".h5", "history")
n_epochs = len(history)
n_epochs = np.arange(1, n_epochs+1)

val_loss = history['val_loss'].values
train_loss = history['loss'].values
val_acc = history['val_acc'].values
train_acc = history['acc'].values

fig = plt.figure()
plt.plot(n_epochs, train_acc, '-b', label='Training')
plt.plot(n_epochs, val_acc, '-r', label='Validation')
plt.title(model_name + ' accuracy', y=1.04)
plt.grid(linestyle=':')
plt.xlabel('Epoch', labelpad=8, fontsize=14)
plt.ylabel('Accuracy', labelpad=10, fontsize=14)
plt.legend(loc='lower right')
plt.savefig(plot_dir + model_name + '_accuracy.pdf', format='pdf')

fig = plt.figure()
plt.plot(n_epochs, train_loss, '-b', label='Training')
plt.plot(n_epochs, val_loss, '-r', label='Validation')
plt.title(model_name + ' loss', y=1.04)
plt.grid(linestyle=':')
plt.xlabel('Epoch', labelpad=8, fontsize=14)
plt.ylabel('Loss', labelpad=10, fontsize=14)
plt.legend(loc='upper right')
plt.savefig(plot_dir + model_name + '_loss.pdf', format='pdf')
