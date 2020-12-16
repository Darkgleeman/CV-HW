# -*- coding: utf-8 -*-
import skimage.io
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import h5py
import cv2
import glob

from CNN import MCNN
from visualize import loss_show
from torch.utils.data import DataLoader
device = 'cuda'

def get_map_pieces(part='partB', train_type='train'):
    path = 'dataset/' + part + '/{}/numpyarrayPiece/*.npy'.format(train_type)
    density_maps = glob.glob(path)
    # density_maps = sorted(density_maps, key = lambda x : int(x[x.index('_')+1 : x.index('.')]))
    density_maps = sorted(density_maps, key = lambda x : sort_name(x))
    density_maps = [np.load(map_name) for map_name in density_maps]
    return density_maps

def sort_name(s):
    s = s.split('/')[-1]
    s = s[:-4]
    key = tuple(s.split('_'))
    return key

def get_img_pieces(part='partB', train_type='train'):
    path = 'dataset/' + part + '/{}/imagePiece/*.jpg'.format(train_type)
    train_imgs = glob.glob(path)
    train_imgs = sorted(train_imgs, key = lambda x : sort_name(x))
    train_imgs = [cv2.imread(img_name) for img_name in train_imgs]
    return train_imgs


class Dataset():
    def __init__(self, part, train_type):
        self.inputs = get_img_pieces(part = part, train_type=train_type)
        self.labels = get_map_pieces(part = part, train_type=train_type)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input = self.inputs[index]
        label = self.labels[index]
        pair = {'input':input, 'label':label}
        return pair

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

def save_info(part, type, epoch, lr, batch_size, net, MSE_running_loss, MAE_running_loss):
    # save the net
    torch.save(net.state_dict(), './model/{}/{}_model_{}_{}_{}.ckpt'.format(part, type, epoch+1, lr, batch_size))
    # save loss
    np.save('./loss/{}/npy/{}model_MSE_{}_{}_{}.npy'.format(part, type, epoch+1, lr, batch_size), MSE_running_loss)
    np.save('./loss/{}/npy/{}model_MAE_{}_{}_{}.npy'.format(part, type, epoch+1, lr, batch_size), MAE_running_loss)
    # visualize loss
    loss_show(MSE_running_loss[5:], './loss/{}/jpg/{}model_MSE_{}_{}_{}.jpg'.format(part, type, epoch+1, lr, batch_size))
    loss_show(MAE_running_loss[5:], './loss/{}/jpg/{}model_MAE_{}_{}_{}.jpg'.format(part, type, epoch+1, lr, batch_size))


def training(net, lr, epoches, batch_size, part, type):
    '''
    @param:
    train_data: the convolved images
    net: the net

    @retrun:
    MSE_running_loss: a list contains all the average MSE loss in one epoch
    MAE_running_loss: a list contains all the average MAE loss in one epoch
    '''

    print("-----------Start Loading data!------")
    print("lr", lr, "epoches", epoches, "batch_size", batch_size, "part", part, "type", type)
    train_data = DataLoader(dataset=Dataset(part=part, train_type='train'), batch_size=batch_size, shuffle=True)
    print("-----------End Loading data!--------")

    print("-----------Start Trainging!---------")
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    MSE_loss_func = torch.nn.MSELoss(reduction='sum')
    MAE_loss_func = torch.nn.L1Loss(reduction='sum')

    MSE_running_loss = []
    MAE_running_loss = []

    numOfImage = len(train_data)

    # epoches is the total running lap
    net.train()
    for epoch in range(epoches):
        
        MSE_running_loss.append(0)
        MAE_running_loss.append(0)

        for i, data in enumerate(train_data):
            print('\rprocessing {}/{}'.format(i, len(train_data)), end='')
            inputs, labels = data['input'], data['label']
            # labels*=(255/torch.max(labels))
            # cv2.imwrite("./labels.jpg", (labels[0]/torch.max(labels[0])).detach().cpu().numpy()*255)
            inputs = torch.autograd.Variable(inputs) 
            # labels = torch.autograd.Variable(labels) 
            # labels = torch.from_numpy(labels)

            # resize
            inputs = inputs.permute(0, 3, 1, 2).type(dtype=torch.float).to(device).contiguous()
            labels = labels.unsqueeze(0)
            labels = labels.permute(1, 0, 3, 2).type(dtype=torch.float).to(device).contiguous()

            # back propagation
            optimizer.zero_grad()
            outputs = net(inputs)
            # print('\nsum1 ',torch.sum(labels[0][0]), labels.shape)
            labels = torch.nn.functional.avg_pool2d(labels, 4, stride=4)
            # print('sum2 ',torch.sum(labels[0][0]), labels.shape)

            MSE_loss = MSE_loss_func(outputs, labels).to(device)
            MAE_loss = MAE_loss_func(outputs, labels).to(device)
            if type == 'MAE':
                MAE_loss.backward()
            else:
                MSE_loss.backward()
            # MAE_loss.backward()

            optimizer.step()

            MSE_running_loss[epoch] += MSE_loss.item()
            MAE_running_loss[epoch] += MAE_loss.item()
            outputs[0][0]*=(255/torch.max(outputs[0][0]))
            labels[0][0]*=(255/torch.max(labels[0][0]))
            cv2.imwrite("./output.jpg", outputs[0][0].detach().cpu().numpy())
            cv2.imwrite("./labels.jpg", labels[0][0].detach().cpu().numpy())
            # torch.cuda.

        # the average loss in one epoch
        MSE_running_loss[epoch] /= numOfImage
        MAE_running_loss[epoch] /= numOfImage
        print("\rEpoch", epoch+1, "MSE loss", MSE_running_loss[epoch])
        print("Epoch", epoch+1, "MAE loss", MAE_running_loss[epoch])

        if (epoch+1) % 5 == 0:
            save_info(part, type, epoch, lr, batch_size, net, MSE_running_loss, MAE_running_loss)
    save_info(part, type, epoch, lr, batch_size, net, MSE_running_loss, MAE_running_loss)
    print("-----------End Trainging!-----------")

