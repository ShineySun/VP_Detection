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

train_data_set = []
test_data_set = []
with open("/mnt/srv/home/tusimple/train/label_data_0313.json") as f:
    for line in f :
        dic = json.loads(line)
        train_data_set.append(dic)
with open("/mnt/srv/home/tusimple/train/label_data_0531.json") as f:
    for line in f :
        dic = json.loads(line)
        train_data_set.append(dic)
        
with open("/mnt/srv/home/tusimple/train/label_data_0601.json") as f:
    for line in f :
        dic = json.loads(line)
        train_data_set.append(dic)
        
with open("/mnt/srv/home/tusimple/test/test_label.json") as f:
    for line in f :
        dic = json.loads(line)
        test_data_set.append(dic)
        
def lane_coordinate(lanes, h_samples):
    coordinates = []
    for i in range(len(lanes)):
        for j in range(len(lanes[i])):
            if lanes[i][j] >= 0:
                coordinates.append([h_samples[j], lanes[i][j]])
    return coordinates

class TusimpleDataset(Dataset):
    def __init__(self, data_set, for_what):
        self.data_set = data_set
        if for_what == 'train': self.path = "/mnt/srv/home/tusimple/train"
        elif for_what == 'test': self.path = "/mnt/srv/home/tusimple/test"
        self.transform = transforms.Compose([
            transforms.ToPILImage(), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406] , std = [0.229, 0.224, 0.225]) # culane데이터
        ])
        
    def __getitem__(self, index):
        data_item = self.data_set[index]
        img = cv2.imread(os.path.join(self.path, data_item["raw_file"]))    
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = self.transform(img_rgb)
        coordinates = lane_coordinate(data_item['lanes'], data_item['h_samples'])
        label = torch.from_numpy(np.array(coordinates, dtype='float32'))
        sample = {
            "data":img_rgb,
            "target":label
        }
        return sample
    
    def __len__(self):
        return len(self.data_set)
    
if __name__ == '__main__':
    train_data = TusimpleDataset(train_data_set, 'train')
    test_data = TusimpleDataset(test_data_set, 'test')
    print(train_data[0])
    print(test_data[0])