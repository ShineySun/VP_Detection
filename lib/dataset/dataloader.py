import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import numpy as np
import os
from os import walk
import cv2
import matplotlib.pyplot as plt

from culane_dataset import CulaneDataset as culane
from culane_dataset import CulaneDatasetTest as culane_test
from tusimple_dataset import TusimpleDataset as tusimple


def make_tusimple_train_valid_loader():
    validation_split = .2
    random_seed= 42
    shuffle_dataset = True
    indices = list(range(len(tusimple('train'))))
    split = int(np.floor(validation_split * len(tusimple('train'))))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        tusimple('train'),
        batch_size = 16,
        sampler = train_sampler
    )
    valid_loader = DataLoader(
        tusimple('train'),
        batch_size=16,
        sampler = valid_sampler
    )
    return train_loader, valid_loader

def make_tusimple_test_loader():
    test_loader = DataLoader(
        tusimple('test'),
        batch_size = 16,
        shuffle = False
    )
    return test_loader

def make_culane_train_valid_loader():
    train_loader = DataLoader(
        culane('train'),
        batch_size = 16,
        shuffle = True,
    )
    valid_loader = DataLoader(
        culane('valid'),
        batch_size = 16,
        shuffle = True,
    )
    return train_loader, valid_loader

def make_culane_test_loader():
    test_loader = DataLoader(
        culane_test(),
        batch_size = 16,
        shuffle = False,
    )
    return test_loader

if __name__ == '__main__':
    print('start')
    train_loader, valid_loader = make_tusimple_train_valid_loader()
    test_loader = make_tusimple_test_loader()
    print(len(train_loader), len(valid_loader), len(test_loader))
    culane_train_loader, culane_valid_loader = make_culane_train_valid_loader()
    culane_test_loader = make_culane_test_loader()
    print(len(culane_train_loader), len(culane_valid_loader), len(culane_test_loader))
