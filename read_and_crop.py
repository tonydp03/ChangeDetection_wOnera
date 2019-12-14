# coding: utf-8

"""
Main functions to preprocess Onera Dataset, available @ http://dase.grss-ieee.org

@Author: Tony Di Pilato

Created on Fri Dec 13, 2019
"""


import os
import numpy as np
from libtiff import TIFF


def get_folderList(txtfile):    
    f = open(txtfile, 'r')
    folders = f.read().split(',')
    return folders

def build_raster(folder):
    filenames = ['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12']
    bands = [TIFF.open(folder + f + '.tif').read_image() for f in filenames]
    raster = np.stack(bands, axis=2)    
    return raster

def pad_raster(raster, crop_size, stride):
    h, w, c = raster.shape
    n_h = int(h/stride)
    n_w = int(w/stride)
    w_extra = w - (n_w * stride)
    w_toadd = crop_size - w_extra
    h_extra = h - (n_h * stride)
    h_toadd = crop_size - h_extra
    img_pad = np.pad(raster, [(0, h_toadd), (0, w_toadd), (0,0)], mode='constant')
    return img_pad

def crop_raster(raster, crop_size, stride):
    cropped_images = []
    h, w, c = raster.shape    
    n_h = int(h/stride)
    n_w = int(w/stride)
    for i in range(n_h):
        for j in range(n_w):
            crop_img = raster[(i * stride):((i * stride) + crop_size), (j * stride):((j * stride) + crop_size), :]
            cropped_images.append(crop_img)
    return cropped_images