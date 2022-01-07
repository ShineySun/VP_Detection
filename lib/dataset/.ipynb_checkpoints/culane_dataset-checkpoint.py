import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import os
from os import walk
import cv2
import matplotlib.pyplot as plt
    
# key point label 출력하기
# coordinates , data = remove_newlines(train_data_set[1]['label_point_path'])
# fig = plt.figure(figsize=(15,10))
# def plot_data (image_file , coordinates):
#     img = cv2.imread(image_file)
#     img_rgb = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
#     for x , y in coordinates :
#         cv2.circle(img_rgb,(int(float(x)), int(float(y)) ) , 5, (0,255,0), -1)
#     plt.imshow(img_rgb)
#     plt.show()
# plot_data(train_data_set[1]['img_path'], coordinates)

# img만 출력하기
# img = cv2.imread(train_data_set[0]['img_path'])
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# print(img.shape)
# plt.imshow(img_rgb)
# plt.show()

# segmentation label  출력하기
# img = cv2.imread(val_data_set[1]['label_path'])
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# print(img.shape)
# ret, thr = cv2.threshold(img, 0.00000001, 255, cv2.THRESH_BINARY) 
# # 값이 너무 작고 적어서 thresh로 차이 벌려줌
# plt.imshow(thr)
# plt.show()

class CulaneDataset(Dataset):
    def __init__(self, for_what='train'):
        self.data_set = []
        if for_what == 'train':
            self._load_train()
        elif for_what == 'valid':
            self._load_valid()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
#             transforms.Resize((590, 1640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406] , std = [0.229, 0.224, 0.225])
        ])
        self.transform_exist = transforms.Compose([
            transforms.ToTensor()
        ])
    def __getitem__(self, index):
        data_item = self.data_set[index]
        img = cv2.imread(data_item["img_path"])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = self.transform(img_rgb)
        coordinates , data = self._remove_newlines(data_item['label_point_path'])
        label = torch.from_numpy(np.array(coordinates, dtype='float32'))
        
        if (len(data_item["exist"]) == 4 and label is not None):
            exists = np.array(data_item["exist"])
        else:
            exists = None
        exists = torch.from_numpy(exists)
        sample = {
            "data":img_rgb,
            "target":label,
            "exist":exists
        }
        return sample
    def __len__(self):
        return len(self.data_set)

    def _load_train(self):
        train_results=[]
        with open("/mnt/srv/home/culane/list/train_gt.txt") as f:
            for line in f :
                line = line.strip()
                data = line.split(" ")
                train_results.append(data)

            for data in train_results:
                exist = [int(x) for x in data[2:]]
            
                img_sep = data[0].split(os.path.sep)
                label_point = data[0].split(os.path.sep)
                label_point[-1] = img_sep[-1][:-3] + 'lines.txt'
            #     label_seg = data[1].split(os.path.sep)

                img_path = os.path.join("/mnt/srv/home/culane/train_set", img_sep[1],img_sep[2],img_sep[3])
                label_point_path = os.path.join("/mnt/srv/home/culane/train_set", label_point[1], label_point[2], label_point[3])
            #     label_seg_path = os.path.join("/mnt/srv/home/culane/laneseg_label_w16/driver_161_90frame", label_seg[-2], label_seg[-1]) # _161_90만 사용
                dict_culane = {
                    "img_path": img_path,
                    "label_point_path" : label_point_path,
            #         "label_seg_path": label_seg_path,
                    "exist": exist
                }
                self.data_set.append(dict_culane)



    def _load_valid(self):
        val_results = []
        with open("/mnt/srv/home/culane/list/val_gt.txt") as f:
            for line in f :
                line = line.strip()
                data = line.split(" ")
                val_results.append(data)
        for data in val_results:
            exist = [int(x) for x in data[2:]]
        
            img_sep = data[0].split(os.path.sep)
            label_point = data[0].split(os.path.sep)
            label_point[-1] = img_sep[-1][:-3] + 'lines.txt'
        #     label_seg = data[1].split(os.path.sep)

            img_path = os.path.join("/mnt/srv/home/culane/train_set", img_sep[1],img_sep[2],img_sep[3])
            label_point_path = os.path.join("/mnt/srv/home/culane/train_set", label_point[1], label_point[2], label_point[3])
        #     label_seg_path = os.path.join("/mnt/srv/home/culane/", label_seg[1], label_seg[2], label_seg[3], label_seg[4])
            dict_culane = {
                "img_path": img_path,
                "label_point_path" : label_point_path,
        #         "label_seg_path": label_seg_path,
                "exist": exist
            }
            self.data_set.append(dict_culane)

    def _remove_newlines(self, fname):
        data=[]
        coordinates =[]
        flist = open(fname).readlines()
        for s in flist:
            data.extend([d for d in s.rstrip('\n').split(" ")  if d!=''])
        data = np.array(data).astype(np.float32)
        for idx in range(0,len(data),2):
            coordinates.append(data[idx:idx+2])
            
        if(len(data)==0):
            return None , None
        else:
            return coordinates , data
    
class CulaneDatasetTest(Dataset):
    def __init__(self):
        self.data_set = []
        self._load_test()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
#             transforms.Resize((590, 1640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406] , std = [0.229, 0.224, 0.225])
        ])
        
    def __getitem__(self, index):
        data_item = self.data_set[index]
        img = cv2.imread(data_item["img_path"])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = self.transform(img_rgb)
        coordinates , data = self._remove_newlines(data_item['label_point_path'])
        label = torch.from_numpy(np.array(coordinates, dtype='float32'))
        
        sample = {
            "data":img_rgb,
            "target":label
        }
        return sample
    def __len__(self):
        return len(self.data_set)
    
    def _load_test(self):
        test_results = []
        with open("/mnt/srv/home/culane/list/test.txt") as f:
            for line in f :
                line = line.strip()
                data = line.split(" ")
                test_results.append(data)
        for data in test_results:
            img_sep = data[0].split(os.path.sep)
            label_point = data[0].split(os.path.sep)
            label_point[-1] = img_sep[-1][:-3] + 'lines.txt'

            img_path = os.path.join("/mnt/srv/home/culane/test_set", img_sep[1],img_sep[2],img_sep[3])
            label_point_path = os.path.join("/mnt/srv/home/culane/test_set", label_point[1], label_point[2], label_point[3])

            dict_culane = {
                "img_path": img_path,
                "label_point_path" : label_point_path,
            }
            self.data_set.append(dict_culane)
    
    def _remove_newlines(self, fname):
        data=[]
        coordinates =[]
        flist = open(fname).readlines()
        for s in flist:
            data.extend([d for d in s.rstrip('\n').split(" ")  if d!=''])
        data = np.array(data).astype(np.float32)
        for idx in range(0,len(data),2):
            coordinates.append(data[idx:idx+2])
            
        if(len(data)==0):
            return None , None
        else:
            return coordinates , data
    
if __name__ == "__main__":
    train_data = CulaneDataset('train')
    val_data = CulaneDataset('valid')
    test_data = CulaneDatasetTest()

    print(test_data[30])
    