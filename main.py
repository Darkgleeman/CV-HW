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
from CNN import MCNN, MCNN_branch1, MCNN_branch2, MCNN_branch3, initialize_MCNN
from line_res_mcnn import line_Res_MCNN, line_Res_MCNN_branch1, line_Res_MCNN_branch2, line_Res_MCNN_branch3
from Res_idea_MCNN import Res_MCNN, Res_MCNN_branch1, Res_MCNN_branch2, Res_MCNN_branch3

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




if __name__ == "__main__":

    device = "cuda"
    net = MCNN().to(device)
    b1 = Res_MCNN_branch1().to(device)
    b2 = Res_MCNN_branch2().to(device)
    b3 = Res_MCNN_branch3().to(device)

    net = Res_MCNN().to(device)
    # net_dict = torch.load('./newModel/partB/MSE_model_100_1e-05_20.ckpt')
    # net.load_state_dict(net_dict)
    
    # # net = MCNN().to(device)

    # part = 'partB'
    # lr = 0.00001
    # training(net, lr=lr, epoches=100, batch_size=20, part=part, type='MSE')
    # # training(net, lr=lr, epoches=100, batch_size=20, part=part, type='MAE')

    # # training(net, lr=lr, epoches=100, batch_size=1, part=part, type='MSE')
    # # training(net, lr=lr, epoches=100, batch_size=1, part=part, type='MAE')

    # # training(net, lr=lr, epoches=100, batch_size=10, part=part, type='MSE')
    # # training(net, lr=lr, epoches=100, batch_size=10, part=part, type='MAE')
    
    # part = 'partA'
    # lr = 0.001
    # training(net, lr=lr, epoches=100, batch_size=20, part=part, type='MSE')
    # training(net, lr=lr, epoches=100, batch_size=20, part=part, type='MAE')

    # training(net, lr=lr, epoches=100, batch_size=1, part=part, type='MSE')
    # training(net, lr=lr, epoches=100, batch_size=1, part=part, type='MAE')
    
    # training(net, lr=lr, epoches=100, batch_size=10, part=part, type='MSE')
    # training(net, lr=lr, epoches=100, batch_size=10, part=part, type='MAE')
    

    # pre train
    # col1 = MCNN_branch1().to(device)
    # col2 = MCNN_branch2().to(device)
    # col3 = MCNN_branch3().to(device)
    # part = 'partB'
    # lr = 0.001
    # # training(col1, lr=lr, epoches = 100, batch_size=20, part=part, type='MSE', id=1)
    # # training(col2, lr=lr, epoches = 100, batch_size=20, part=part, type='MSE', id=2)
    # training(col3, lr=lr, epoches = 100, batch_size=20, part=part, type='MSE')


    # train the whole model with pre-trained columns
    # net = MCNN().to(device)
    # d1 = torch.load('./model/partB/pretrain1/MSE_model_70_0.001_20.ckpt')
    # d2 = torch.load('./model/partB/pretrain2/MSE_model_45_0.001_20.ckpt')
    # d3 = torch.load('./model/partB/pretrain3/MSE_model_55_0.001_20.ckpt')
    # dict = initialize_MCNN(d1, d2, d3, net)
    # net.load_state_dict(dict)
    part = 'partB'
    lr = 0.0001
    training(net, lr=lr, epoches=100, batch_size=20, part=part, type='MSE')