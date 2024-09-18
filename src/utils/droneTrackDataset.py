"""
author: alfeak
time: 2024-09-17
description:
    custom dataset for drone tracking
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class dronetrack_dataset(Dataset):
    def __init__(self,data_root_path,train=True):
        super(dronetrack_dataset,self).__init__()
        self.data_root_path = data_root_path
        self.data = self.load_data()
        self.train = train
    def load_data(self):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

class event2_dataset(dronetrack_dataset):
    def __init__(self,data_root_path):
        super(event2_dataset,self).__init__(data_root_path)

    def load_data(self):
        return(np.load(self.data_root_path))

    def __getitem__(self, index):
        data_t = self.data[index]
        data = torch.from_numpy(data_t[:-1,1:-1]).float()
        label = torch.from_numpy(data_t[-1])[0].float() #
        return data,label

    def __len__(self):
        return self.data.shape[0]

if __name__ == '__main__':
    data_root_path = 'pre_expriment/eval/val.npy'
    dataset = event2_dataset(data_root_path)
    dataloader = DataLoader(dataset,batch_size=32,shuffle=True)
    for data,label in dataloader:
        print(data.shape,label.shape)
