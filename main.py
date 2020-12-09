import glob
import torch
import cv2
import numpy as np

import h5py
import os
import skimage.io as io

import model
import train
import CNN


def get_density_maps():
    h5_names = glob.glob('dataset/train/ground-truth-h5/*.h5')
    # sort in increasing order
    h5_names = sorted(h5_names, key=lambda x : int(x[x.index('_')+1 : x.index('.')]))
    # desity_maps with shape(1024, 768), np.array
    maps = []
    for h5 in h5_names:
        h5 = h5py.File(h5, 'r')['density']
        map = np.array(h5).T # transpose it to get the consistent size
        maps.append(map)
    return maps


def get_imgs():
    train_imgs = glob.glob("dataset/train/imgs/*.jpg")
    train_imgs = sorted(train_imgs, key=lambda x : int(x[x.index('_')+1 : x.index('.')]))
    imgs = [cv2.imread(img_name) for img_name in train_imgs]
    return imgs


def pre_processor():
    density_maps = get_density_maps()
    imgs = get_imgs()
    splited_img = [[],[],[]]
    splited_density = [[],[],[]]
    h = 512
    w = 384
    for raw_density, raw_img in zip(density_maps, imgs):
        for i in range(3):
            for j in range(3):
                splited_img[i].append(raw_img[h//2*j:h//2*(j+2), w//2*i:w//2*(i+2)]) #split into 9 parts
                splited_density[i].append(raw_density[h//2*j:h//2*(j+2),w//2*i:w//2*(i+2)])
        yield splited_img , splited_density
    
    

if __name__ == "__main__":
    epoch = 100
    for i in range(epoch):
        batch_iterater = pre_processor()
        for patch_to_train , ground_truth in batch_iterater:
            pass
