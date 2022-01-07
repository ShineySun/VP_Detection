import numpy as np
from scipy import io
import os
import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt


class VPGDataset(Dataset):
    def __init__(self, for_what='train'):
        self.path = '/mnt/srv/home/vpgnet_dataset'
        self.files = self._get_file_names(for_what)

        self.transform = transforms.Compose([
            transforms.ToPILImage(), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406] , std = [0.229, 0.224, 0.225]) # culane데이터
        ])

    def __getitem__(self, index):
        rgb_img, label, vp = self._load_data(self.files[index])
        rgb_img = np.squeeze(rgb_img)
        rgb_img = self.transform(rgb_img)
        label = torch.from_numpy(label.squeeze().T)
        vp = torch.from_numpy(vp.squeeze())
        sample = {
            "data":rgb_img,
            "label":label,
            "vp":vp,
        }
        return sample
    def __len__(self):
        return len(self.files)

    def _get_file_names(self, for_what='train'):
        files = []
        if for_what == 'train':
            scene_list = ['scene_1/*', 'scene_2/*']
        elif for_what == 'valid':
            scene_list = ['scene_3/*']
        else: scene_list = ['scene_4/*']
        
        for i in scene_list: # train file name list
            path_scene = os.path.join(self.path, i)
            tmp = [f+'/*' for f in glob.glob(path_scene)]
            for f in tmp:
                files.extend(glob.glob(f))
        return files

    def _load_data(self, file):
        label_seg = []
        vp_gt = []
        rgb_img = []

        mat_file = io.loadmat(file)
        data = mat_file['rgb_seg_vp']
        rgb_img.append(data[:,:,:3])
        label_seg_tmp = data[:,:,3]
        label_seg.append(np.where((label_seg_tmp==1) | (label_seg_tmp==2) | (label_seg_tmp==4) | (label_seg_tmp==5) | (label_seg_tmp==7)))
        vp_gt_tmp = data[:,:,4]
        vp_gt.append(np.where(vp_gt_tmp>=1))
            
        rgb_img = np.array(rgb_img)
        label_seg = np.array(label_seg)
        vp_gt = np.array(vp_gt)
        return rgb_img, label_seg, vp_gt

if __name__ == '__main__':
    train_set = VPGDataset('train')
    valid_set = VPGDataset('valid')
    test_set = VPGDataset('test')
    print(len(valid_set))
    one = valid_set[0]
    img = one['data']
    lane = one['label']
    vp = one['vp']
    print(img.shape, lane.shape, vp.shape)