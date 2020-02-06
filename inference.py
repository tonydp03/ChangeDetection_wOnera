# coding: utf-8

"""
Perform inference with a CD UNet/UNet++ model for Onera Dataset, available @ http://dase.grss-ieee.org

@Author: Tony Di Pilato

Created on Fri Dec 13, 2019
"""

import os
import read_and_crop as rnc
import numpy as np
from libtiff import TIFF
import tensorflow as tf
import cd_models as cdm
from keras.models import load_model
import matplotlib.pyplot as plt
import random


img_size = 256
channels = 13
stride = 256
classes = 1

dataset_dir = '../OneraDataset_Images/'
labels_dir = '../OneraDataset_TrainLabels/'

save_dir = '../models/'
model_name = 'EF_UNet_bce-256_ol64'

infres_dir = '../results/'
history_name = model_name + '_history'

# Get the list of folders to open to get rasters
# folders = rnc.get_folderList(dataset_dir + 'test.txt')
folders = rnc.get_folderList(dataset_dir + 'train.txt')

# Select a folder, build raster, pad it and crop it to get the input images
# f = random.choice(folders)
f = 'rennes'

raster1 = rnc.build_raster(dataset_dir + f + '/imgs_1_rect/')
raster2 = rnc.build_raster(dataset_dir + f + '/imgs_2_rect/')
raster = np.concatenate((raster1,raster2), axis=2)
padded_raster = rnc.pad(raster, img_size)
test_image = rnc.crop(padded_raster, img_size, stride)

# Create inputs for the Neural Network
inputs = np.asarray(test_image)

# Load model
model = load_model(save_dir + model_name + '.h5', custom_objects={'weighted_bce_dice_loss': cdm.weighted_bce_dice_loss})
# model = load_model(save_dir + model_name + '.h5')
model.summary()

print("Model loaded!")

# Perform inference
results = model.predict(inputs)

# Build the complete change map
# results = results[4] # This should be used if DS enabled
shape = (padded_raster.shape[0], padded_raster.shape[1], classes)
padded_cm = rnc.uncrop(shape, results, img_size, stride)
cm = rnc.unpad(raster.shape, padded_cm)

cm = np.squeeze(cm)
cm = np.rint(cm) # we are only interested in change/unchange

res_dir = infres_dir + f

if not os.path.exists(res_dir):
    os.mkdir(res_dir)

# Plot and save the change map
fig = plt.imshow(cm, cmap='gray')
plt.axis('off')
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.savefig(res_dir + '/' + model_name + '.png', bbox_inches = 'tight', pad_inches = 0)