import glob
import torch
import cv2
import h5py
import numpy as np
import skimage.io as io
from torch.utils.data import DataLoader
################### local module ######################

import model
import train
import CNN
from train import training
from CNN import MCNN

######################################################

def get_density_maps():
    h5_names = glob.glob('dataset/train/ground-truth-h5/*.h5')
    h5_names = sorted(h5_names, key = lambda x : int(x[x.index('_')+1 : x.index('.')]))# sort in increasing order
    density_maps = []# desity_maps with shape(1024, 768), np.array
    for h5 in h5_names:
        h5 = h5py.File(h5, 'r')['density']
        density_map = np.array(h5).T # transpose it to get the consistent size
        density_maps.append(density_map)
    return density_maps


def get_imgs():
    train_imgs = glob.glob("dataset/train/imgs/*.jpg")
    train_imgs = sorted(train_imgs, key=lambda x : int(x[x.index('_')+1 : x.index('.')]))
    imgs = [cv2.imread(img_name) for img_name in train_imgs]
    return imgs


def pre_processor():    #it is a generator
    density_maps = get_density_maps()
    imgs         = get_imgs()
    splited_img  = [[],[],[]]
    splited_density = [[],[],[]]
    h = 512  # half the size
    w = 384
    splited_img     = np.ndarray((9,3,h,w))
    splited_density = np.ndarray((9,3,h,w))
    for raw_density, raw_img in zip(density_maps, imgs):
        for i in range(3):
            for j in range(3):
                splited_img[i*3+j]     = raw_img[:,h//2*j:h//2*(j+2), w//2*i:w//2*(i+2)]   #split into 9 parts
                splited_density[i*3+j] = raw_density[:,h//2*j:h//2*(j+2),w//2*i:w//2*(i+2)] #dimension may be wrong
        yield splited_img , splited_density

def sort_name(s):
    s = s.split('/')[-1]
    s = s[:-4]
    key = tuple(s.split('_'))
    return key
    

def get_map_pieces(part='partA'):
    path = 'dataset/' + part + '/train/numpyarrayPiece/*.npy'
    density_maps = glob.glob(path)
    # density_maps = sorted(density_maps, key = lambda x : int(x[x.index('_')+1 : x.index('.')]))
    density_maps = sorted(density_maps, key = lambda x : sort_name(x))
    density_maps = [np.load(map_name) for map_name in density_maps]
    return density_maps


def get_img_pieces(part='partA'):
    path = 'dataset/' + part + '/train/imagePiece/*.jpg'
    train_imgs = glob.glob(path)
    # train_imgs = sorted(train_imgs, key = lambda x : int(x[x.index('_')+1 : x.index('.')]))
    train_imgs = sorted(train_imgs, key = lambda x : sort_name(x))
    train_imgs = [cv2.imread(img_name) for img_name in train_imgs]
    return train_imgs


class Dataset():
    def __init__(self, type):
        self.inputs = get_img_pieces()
        self.labels = get_map_pieces()
        self.type = type

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input = self.inputs[index]
        label = self.labels[index]
        pair = {'input':input, 'label':label}
        return pair
        
# if __name__ == "__main__":
#     epoch = 100
#     for i in range(epoch):
#         batch_iterater = pre_processor()
#         for patch_to_train , ground_truth in batch_iterater:# 2 numpy-array 
#             pass

if __name__ == "__main__":

    device = "cuda"

    net = MCNN().to(device)

    # hyperparameters
    batch_size = 1
    lr = 0.01
    epoch_num = 100

    train_data = DataLoader(dataset=Dataset('train'), batch_size=batch_size, shuffle=True)

    training(train_data, net)