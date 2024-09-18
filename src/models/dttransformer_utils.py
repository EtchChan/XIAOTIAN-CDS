"""
author : alfeak
date : 2024-09-17
description : 
    model utils functions and transformer implementation for dtransformer
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class res_layer(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=3,stride=2,padding=1):
        super(res_layer, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3,stride=stride, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm1d(out_channels),
        )
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=3,stride=stride, padding=1)
    
    def forward(self, x):
        out = self.conv_layer(x)
        res = self.residual(x)
        out += res
        out = F.relu(out)
        return out
    