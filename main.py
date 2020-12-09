import glob
import torch
import cv2
import numpy as np

import h5py
import os
import skimage.io as io



def get_paths(directory):
    train_imgs = glob.glob("dataset/train/img/*.jpg")
    train_density = glob.glob("dataset/train/density_maps/*.jpg")
    return (train_imgs,train_density)


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


def pre_processor(img_path,density_path):
    density_maps = get_density_maps()
    imgs = get_imgs()
    splited_img = [[],[],[]]
    splited_density = [[],[],[]]
    for raw_density, raw_img in zip(density_maps, imgs):
        h = 512
        w = 384
        for i in range(3):
            for j in range(3):
                splited_img[i].append(raw_img[h//2*j:h//2*(j+2), w//2*i:w//2*(i+2)]) #split into 9 parts
                splited_density[i].append(raw_density[h//2*j:h//2*(j+2),w//2*i:w//2*(i+2)])
    

if __name__ == "__main__":
    pre_processor(None, None)