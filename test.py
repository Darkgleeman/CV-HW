# -*- coding: utf-8 -*-
import skimage.io
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import h5py
from torch.utils.data import DataLoader
import cv2
import glob
################### local module ######################

import model
import train
import CNN
from CNN import MCNN
from train import training, Dataset
from visualize import loss_show

######################################################


if __name__ == '__main__':

    batch_size = 1
    device = "cuda"
    net = MCNN().to(device)
    net_dicts = glob.glob('./model/partB/MSE_model_*_1e-05_20.ckpt')
    net_dicts = sorted(net_dicts, key=lambda x : int(x.split('_')[2]))
    epoches = [int(x.split('_')[2]) for x in net_dicts]


    MSE_loss_func = torch.nn.MSELoss(reduction='sum')
    MAE_loss_func = torch.nn.L1Loss(reduction='sum')

    MSE_running_loss = []
    MAE_running_loss = []
    test_data = DataLoader(dataset=Dataset('partA', 'test'), batch_size=batch_size, shuffle=False)
    for i, net_dict in enumerate(net_dicts):
        net_dict = torch.load(net_dict)
        net.eval()
        net.load_state_dict(net_dict)

        print("Start Testing!")

        MSE_running_loss.append(0)
        MAE_running_loss.append(0)

        numOfImage = len(test_data)

        for j, data in enumerate(test_data):
            print('\rprocessing {}/{}'.format(j, len(test_data)), end='')
            inputs, labels = data['input'], data['label']
            cv2.imwrite("./input.jpg", inputs[0][0].detach().cpu().numpy())

            # resize
            inputs = inputs.permute(0, 3, 1, 2).type(dtype=torch.float).to(device) 
            labels = labels.permute(0, 2, 1).unsqueeze(0).type(dtype=torch.float).to(device)

            outputs = net(inputs)
            
            # resize the labels
            labels = torch.nn.functional.avg_pool2d(labels, 4, stride=4)
            # print(torch.sum(outputs[0][0]).item(), torch.sum(labels[0]).item())

            MSE_loss = MSE_loss_func(outputs, labels)
            MAE_loss = MAE_loss_func(outputs, labels)

            MSE_running_loss[i] += MSE_loss.item()
            MAE_running_loss[i] += MAE_loss.item() * 16

            outputs[0][0]*=(255/torch.max(outputs[0][0]))
            labels[0][0]*=(255/torch.max(labels[0][0]))
            
            cv2.imwrite("./output.jpg", outputs[0][0].detach().cpu().numpy())
            cv2.imwrite("./labels.jpg", labels[0][0].detach().cpu().numpy())
        # the average loss in one epoch
        MSE_running_loss[i] /= numOfImage
        MAE_running_loss[i] /= numOfImage
        print("\rMSE loss", MSE_running_loss[i])
        print("MAE loss", MAE_running_loss[i])
        print("End Testing!")
        lr = 0.00001
        loss_show(MSE_running_loss, "./loss/test/partB/MSE_{}.jpg".format(lr), epoches[:i+1])
        loss_show(MAE_running_loss, "./loss/test/partB/MAE_{}.jpg".format(lr), epoches[:i+1])

