"""
author : alfeak
date : 2024-09-18
description : 
    backbone for drone tracking classification
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from dttransformer_utils import res_layer

class get_model(nn.Module):
    def __init__(self,in_channel=7,num_class=2,is_train=True):
        super(get_model, self).__init__()
        self.num_class = num_class
        self.is_train = is_train
        self.in_channel = in_channel
        self.embedding = nn.Sequential(
            nn.Conv1d(in_channel, 96, kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
        )
        self.layer = nn.Sequential(
            res_layer(96, 128),
            res_layer(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_class)
    def forward(self, x):
        B, N, C = x.shape
        x = x.permute(0, 2, 1)
        x = self.embedding(x)
        # x = x.permute(0, 2, 1)
        x = self.layer(x)
        # x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze()
        x = self.fc(x)
        return x


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.cross_entropy(pred, target)

        return total_loss
    
if __name__ == '__main__':
    model = get_model()
    print(model)
    model = model.train()
    data = torch.randn(4, 15, 7)
    out = model(data)
    print(out.shape)