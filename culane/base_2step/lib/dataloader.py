# -*- coding: utf-8 -*-
from re import X
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import os
from os import walk
import cv2
import matplotlib.pyplot as plt
import json


class CulaneDataset(Dataset):
    def __init__(self, for_what='train'):
        self.for_what = for_what
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            #             transforms.Resize((590, 1640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]) # imagenet 
        ])
        if for_what == 'train':
            self.train_lst = []
            with open('../data/culane_vp_gt/train_culane_curve_1d_hlimit_0.json', 'rb') as f:
                data = json.load(f)
                for i in data['data']:
                    if i['annotations']['num_lanes'] < 2: continue
                    tmp = i['metadata']['img_info']
                    tmp.update(i['annotations']['mean_vp'])
                    tmp['num_lanes'] = i['annotations']['num_lanes']
                    self.train_lst.append(tmp)
            # print(len(self.train_lst))

            self.valid_lst = []
            with open('../data/culane_vp_gt/val_culane_curve_1d_hlimit_0.json', 'rb') as f:
                data = json.load(f)
                for i in data['data']:
                    if i['annotations']['num_lanes'] < 2: continue
                    tmp = i['metadata']['img_info']
                    tmp.update(i['annotations']['mean_vp'])
                    tmp['num_lanes'] = i['annotations']['num_lanes']
                    self.valid_lst.append(tmp)
            # print(len(self.valid_lst))
        elif for_what == 'test':
            self.test_lst = []
            with open('../data/culane_vp_gt/test_culane_curve_1d_hlimit_0.json', 'rb') as f:
                data = json.load(f)
                for i in data['data']:
                    if i['annotations']['num_lanes'] < 2: continue
                    tmp = i['metadata']['img_info']
                    tmp.update(i['annotations']['mean_vp'])
                    tmp['num_lanes'] = i['annotations']['num_lanes']
                    self.test_lst.append(tmp)
            # print(len(self.test_lst))
    

    def __getitem__(self, index):
        if self.for_what == 'train':
            self.data_item = self.train_lst[index]
        elif self.for_what == 'test':
            self.data_item = self.test_lst[index]
            print(self.data_item)

            path_img = os.path.join("../data", self.data_item["image_path"])
            path_label = os.path.join("../data", self.data_item["label_path"])
            img = cv2.imread(path_img)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # cv2.imshow('ww', img_rgb)
            # cv2.waitKey(0)
            img_rgb = self.transform(img_rgb)
            # print(img_rgb)
            coordinates = self._remove_newlines(path_label)
            print(coordinates)
        
        return self.data_item

    def _remove_newlines(self, fname):
        coordinates = []
        flist = open(fname).readlines()
        for s in flist:
            tmp0 = ''
            tmp = []
            for i, d in enumerate(s.rstrip('\n').split(" ")):
                if i % 2 == 0: #x좌표
                    tmp0 = d
                else: #y좌표
                    tmp.append((tmp0, d))
                    tmp0 = ''
            coordinates.append(tmp)
        if(len(coordinates) == 0):
            return None
        else:
            return coordinates


if __name__ == '__main__':
    dataset = CulaneDataset('test')
    d = dataset[0]
