import torch
from torch import nn, optim
import numpy as np
from config import device
import torch.nn.functional as F

class Q_Network_Conv(nn.Module):
    def __init__(self, nc=1, ngf = 16):
        super(Q_Network_Conv, self).__init__()
        self.ngf = ngf
        self.conv = nn.Sequential(
            nn.Conv2d( nc, ngf, 5, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.SELU(True),

            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.SELU(True),

            nn.Conv2d( ngf, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.SELU(True)
        )

        self.predict = nn.Sequential(nn.Linear(ngf + 4, 1), nn.Tanh())




    def forward(self, x, action, connect_x=torch.ones(1).to(device) * 4):

        config = torch.unsqueeze(torch.cat((torch.ones(1).to(device) * x.shape[2], torch.ones(1).to(device) * x.shape[3], action, connect_x), dim=0), dim=0)
        config = config.repeat(x.shape[0], 1)
        x = self.conv(x.float())
        x = x.reshape(x.shape[0], self.ngf, -1)
        x, _ = torch.max(x, dim=2)
        x = self.predict(torch.cat((x, config), dim=1))

        return x

class Q_Network_Flat(nn.Module):
    def __init__(self, nc=6*7, ngf = 128, actions=7):
        super(Q_Network_Flat, self).__init__()
        self.ngf = ngf
        self.predict = nn.Sequential(nn.Linear(nc, ngf), nn.SELU(),
                                    nn.Linear(ngf, ngf), nn.SELU(),
                                    nn.Linear(ngf, actions))#, nn.Tanh())




    def forward(self, x):
        x = x.reshape(x.shape[0], -1).float()
        x = self.predict(x)

        return x

class Q_Network(nn.Module):
    def __init__(self, ngf = 64, actions=7):
        super(Q_Network, self).__init__()
        self.ngf = ngf
        self.conv = nn.Sequential(
            nn.Conv2d(3, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.PReLU(),

            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.PReLU(),

            nn.Conv2d( ngf, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.PReLU()
        )

        self.predict = nn.Sequential(nn.Linear(126, ngf), nn.SELU(),
                                    nn.Linear(ngf, ngf), nn.SELU(),
                                    nn.Linear(ngf, actions))#, nn.Tanh())

    def forward(self, x):
        rows = x.shape[2]
        columns = x.shape[3]
        x_new = torch.empty(x.shape[0], 3, rows, columns).to(device)
        for i in range(x.shape[0]):
            x_new[i] = F.one_hot(x[i].reshape(rows, columns).long(), 3).permute(2,0,1)

        x = x_new
        x = x.reshape(x.shape[0], -1).float()
        x = self.predict(x)

        return x