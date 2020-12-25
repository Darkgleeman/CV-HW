import torch
import torch.nn as nn
import copy
import cv2
import numpy as np
from torchvision import transforms

device = 'cuda'
def rgb2gray(shape, tensor):
    shape = (shape[3], shape[2])
    t = np.zeros((tensor.shape[0], 1, shape[1], shape[0]))
    t = torch.from_numpy(t).type(dtype=torch.float).to(device).contiguous()
    for i, img in enumerate(tensor):
        img = img.permute(1, 2, 0)
        img = img.detach().cpu().numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, shape)
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)
        t[i] = img
    return t

class Res_MCNN_branch1(nn.Module):
    def __init__(self):
        super(Res_MCNN_branch1,self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 16, 9, padding=4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 16, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(16, 8, 7, padding=3),
            nn.ReLU()
        )
        self.relu = nn.Sequential(nn.ReLU())
        self.branch1_conv1x1 = nn.Sequential(nn.Conv2d(8,1,1,padding=0))

    def forward(self,input):
        x = self.branch1(input)
        x = self.branch1_conv1x1(x)
        #x is 1-dimension,we have to make input 1-d and its size equals to x
        modified_input = rgb2gray(x.shape, input)
        return self.relu(x+self.relu(modified_input))

class Res_MCNN_branch2(nn.Module):
    def __init__(self):
        super(Res_MCNN_branch2,self).__init__()
        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 20, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 40, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(40, 20, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(20, 10, 5, padding=2),
            nn.ReLU()
        )
        self.relu = nn.Sequential(nn.ReLU())
        self.branch2_conv1x1 = nn.Sequential(nn.Conv2d(10,1,1,padding=0))

    def forward(self,input):
        x = self.branch2(input)
        x = self.branch2_conv1x1(x)
        modified_input = rgb2gray(x.shape, input)
        return self.relu(x+self.relu(modified_input))

class Res_MCNN_branch3(nn.Module):
    def __init__(self):
        super(Res_MCNN_branch3,self).__init__()
        self.branch3 = nn.Sequential(
            nn.Conv2d(3, 24, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 24, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 12, 3, padding=1),
            nn.ReLU()
        )
        self.relu = nn.Sequential(nn.ReLU())
        self.branch3_conv1x1 = nn.Sequential(nn.Conv2d(12,1,1,padding=0))

    def forward(self,input):
        x = self.branch3(input)
        x = self.branch3_conv1x1(x)
        modified_input = rgb2gray(x.shape, input)
        return self.relu(x+self.relu(modified_input))


class Res_MCNN(nn.Module):
    def __init__(self):
        super(Res_MCNN,self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 16, 9, padding=4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 16, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(16, 8, 7, padding=3),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 20, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 40, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(40, 20, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(20, 10, 5, padding=2),
            nn.ReLU()
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(3, 24, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 24, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 12, 3, padding=1),
            nn.ReLU()
        )
        self.relu = nn.Sequential(nn.ReLU())
        self.merge = nn.Sequential(nn.Conv2d(30,1,1,padding=0))

    def forward(self,input):
        x1 = self.branch1(input)
        x2 = self.branch2(input)
        x3 = self.branch3(input)
        x = torch.cat((x1,x2,x3),1)
        x = self.merge(x)

        modified_input = rgb2gray(x.shape, input)

        return self.relu(x+self.relu(modified_input))

def initialize_MCNN(b1_state_dict, b2_state_dict, b3_state_dict, net):
    # b1_state_dict = branch1_model.state_dict()
    # b2_state_dict = branch2_model.state_dict()
    # b3_state_dict = branch3_model.state_dict()
    MCNN_state_dict = net.state_dict()
    new1_state_dict = {key:tensor for key,tensor in b1_state_dict.items() if key in MCNN_state_dict}
    MCNN_state_dict.update(new1_state_dict)
    new2_state_dict = {key: tensor for key, tensor in b2_state_dict.items() if key in MCNN_state_dict}
    MCNN_state_dict.update(new2_state_dict)
    new3_state_dict = {key: tensor for key, tensor in b3_state_dict.items() if key in MCNN_state_dict}
    MCNN_state_dict.update(new3_state_dict)
    return MCNN_state_dict