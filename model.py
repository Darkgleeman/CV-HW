import torch
import torch.nn as nn
import copy

class MCNN_branch1(nn.Module):
    def __init__(self):
        super(MCNN_branch1,self).__init__()
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
        self.branch1_conv1x1 = nn.Sequential(nn.Conv2d(8,1,1,padding=0))

    def forward(self,input):
        x = self.branch1(input)
        x = self.branch1_conv1x1(x)
        return x

class MCNN_branch2(nn.Module):
    def __init__(self):
        super(MCNN_branch2,self).__init__()
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
        self.branch2_conv1x1 = nn.Sequential(nn.Conv2d(10,1,1,padding=0))

    def forward(self,input):
        x = self.branch2(input)
        x = self.branch2_conv1x1(x)
        return x

class MCNN_branch3(nn.Module):
    def __init__(self):
        super(MCNN_branch3,self).__init__()
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
        self.branch3_conv1x1 = nn.Sequential(nn.Conv2d(12,1,1,padding=0))

    def forward(self,input):
        x = self.branch1(input)
        x = self.branch1_conv1x1(x)
        return x

class MCNN(nn.Module):
    def __init__(self,branch1_model,branch2_model,branch3_model):
        super(MCNN,self).__init__()
        self.branch1 = copy.deepcopy(branch1_model)
        self.branch2 = copy.deepcopy(branch2_model)
        self.branch3 = copy.deepcopy(branch3_model)
        nn.init.normal_(self.branch1_conv1x1.weight.data)
        nn.init.normal_(self.branch1_conv1x1.bias.data)
        nn.init.normal_(self.branch2_conv1x1.weight.data)
        nn.init.normal_(self.branch2_conv1x1.bias.data)
        nn.init.normal_(self.branch3_conv1x1.weight.data)
        nn.init.normal_(self.branch3_conv1x1.bias.data)


    def forward(self,input):
        x1 = self.branch1(input)
        x2 = self.branch2(input)
        x3 = self.branch3(input)
        x = x1+x2+x3
        return x


