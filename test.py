# -*- coding: utf-8 -*-
import skimage.io
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import h5py
import cv2
from torch.utils.data import DataLoader
################### local module ######################

import model
import train
import CNN
from CNN import MCNN
from train import training, save_MCNN, load_MCNN
from main import Dataset
from visualize import loss_show

######################################################


if __name__ == '__main__':
    epoch_num = 9
    batch_size = 1
    device = "cpu"
    net = MCNN().to(device)
    load_MCNN('./model/train/train_model_35_0.01.h5', net)
    test_set = Dataset('test')
    test_set.inputs = [cv2.imread("1_0.jpg")]
    test_set.labels = [None]
    test_data = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

    print("Start Testing!")
    MSE_loss_func = torch.nn.MSELoss(reduce = True, size_average = True)
    MAE_loss_func = torch.nn.L1Loss(reduce = True, size_average = True)

    MSE_running_loss = []
    MAE_running_loss = []
    num_loss = np.ndarray((epoch_num))
    for epoch in range(epoch_num):
        
        MSE_running_loss.append(0)
        MAE_running_loss.append(0)

        for i, data in enumerate(test_data):
            print('\rprocessing {}/{}'.format(i, len(test_data)), end='')
            inputs, labels = data['input'], data['label']
            inputs = torch.autograd.Variable(inputs) 
            labels = torch.autograd.Variable(labels) 

            # resize
            inputs = inputs.permute(0, 3, 1, 2).type(dtype=torch.float).to(device) 
            labels = labels.permute(0, 2, 1).unsqueeze(0).type(dtype=torch.float).to(device)

            outputs = net(inputs)
            # resize the labels
            labels.resize_(outputs.shape)

            MSE_loss = MSE_loss_func(outputs, labels)
            MAE_loss = MAE_loss_func(outputs, labels)
            num_loss[epoch] = abs(outputs.sum() - labels.sum())

            MSE_running_loss[epoch] += MSE_loss.item()
            MAE_running_loss[epoch] += MSE_loss.item()
    #save_MCNN('./model/test/test_model.h5', net)
    print("End Trainging!")

    loss_show(num_loss,'counting loss')
