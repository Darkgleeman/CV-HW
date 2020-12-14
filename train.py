# -*- coding: utf-8 -*-
import skimage.io
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import h5py


from CNN import MCNN
from visualize import loss_show
device = 'cuda'

def save_MCNN(h5_save_path, net):
    '''
    @param:
    h5_save_path: the path that you want to save the net in .h5 file
    net: the net 

    @retrun:
    no retrun
    '''
    h5_file = h5py.File(h5_save_path, mode = 'w')

    for key, value in net.state_dict().items():
        h5_file.create_dataset(key, data = value.cpu().numpy())


def load_MCNN(h5_save_path, net):
    '''
    @param:
    h5_save_path: the path that you saved net in .h5 file
    net: the net 

    @retrun:
    no retrun
    '''

    # net is a reference 
    h5_file = h5py.File(h5_save_path, mode = 'w')
    for key, value in net.state_dict().items():
        param = torch.from_numpy(np.asarray(h5_file[key]))
        value.copy_(param)


def training(train_data, net):
    '''
    @param:
    train_data: the convolved images
    net: the net

    @retrun:
    MSE_running_loss: a list contains all the average MSE loss in one epoch
    MAE_running_loss: a list contains all the average MAE loss in one epoch
    '''
    lr = 0.01
    print("Start Trainging!")
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    MSE_loss_func = torch.nn.MSELoss(reduce = True, size_average = True)
    MAE_loss_func = torch.nn.L1Loss(reduce = True, size_average = True)

    MSE_running_loss = []
    MAE_running_loss = []

    numOfImage = len(train_data)

    # epoches is the total running lap
    epoches = 10
    for epoch in range(epoches):
        
        MSE_running_loss.append(0)
        MAE_running_loss.append(0)

        for i, data in enumerate(train_data):
            print('\rprocessing {}/{}'.format(i, len(train_data)), end='')
            inputs, labels = data['input'], data['label']
            inputs = torch.autograd.Variable(inputs) 
            labels = torch.autograd.Variable(labels) 

            # resize
            inputs = inputs.permute(0, 3, 1, 2).type(dtype=torch.float).to(device) 
            labels = labels.permute(0, 2, 1).unsqueeze(0).type(dtype=torch.float).to(device)

            # back propagation
            optimizer.zero_grad()
            outputs = net(inputs)

            # resize the labels
            labels.resize_(outputs.shape)

            MSE_loss = MSE_loss_func(outputs, labels).to(device)
            MAE_loss = MAE_loss_func(outputs, labels).to(device)

            MSE_loss.backward(retain_graph=True)
            MAE_loss.backward()

            optimizer.step()

            MSE_running_loss[epoch] += MSE_loss.item()
            MAE_running_loss[epoch] += MSE_loss.item()

        # the average loss in one epoch
        MSE_running_loss[epoch] /= numOfImage
        MAE_running_loss[epoch] /= numOfImage
        print("\rEpoch", epoch+1, "MSE loss", MSE_running_loss[epoch])
        print("Epoch", epoch+1, "MAE loss", MAE_running_loss[epoch])
    if (epoch+1) % 5 == 0:
        save_MCNN('./model/train/train_model_{}_{}.h5'.format(epoch+1, lr), net)

    save_MCNN('./model/train/train_model_{}_{}.h5'.format(epoch+1, lr), net)
    loss_show(MSE_running_loss, "MSE loss")
    loss_show(MAE_running_loss, "MAE loss")

    print("End Trainging!")

    return MSE_running_loss, MAE_running_loss

