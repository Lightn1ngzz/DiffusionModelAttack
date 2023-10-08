import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms as T
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
import random
 
class hand_pose(Dataset):
    def __init__(self, root, train=True, transforms=None):
        imgs = []
        for path in os.listdir(root):
            path_prefix = path[:3]
            if path_prefix == '000' :
                label = 0
            elif path_prefix == '001':
                label = 1
            elif path_prefix == '002':
                label = 2
            elif path_prefix == '003':
                label = 3
            elif path_prefix == '004':
                label = 4
            elif path_prefix == '005':
                label = 5
            elif path_prefix == '006':
                label = 6
            elif path_prefix == '007':
                label = 7
            elif path_prefix == '008':
                label = 8
            elif path_prefix == '009':
                label = 9
            elif path_prefix == '010':
                label = 10
            elif path_prefix == '011':
                label = 11
            elif path_prefix == '012':
                label = 12
            elif path_prefix == '013':
                label = 13
            else:
                print("data label error")
            #  print(label, '!!!!!')
            childpath = os.path.join(root, path)
            for imgpath in os.listdir(childpath):
                imgs.append((os.path.join(childpath, imgpath), label))
        
        train_path_list, val_path_list = self._split_data_set(imgs)
        if train:
            self.imgs = train_path_list
        else:
            self.imgs = val_path_list
 
        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
 
            self.transforms = T.Compose([
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
            ])
        else:
            self.transforms = transforms
             
    def __getitem__(self, index):
        img_path = self.imgs[index][0]
        label = self.imgs[index][1]
 
        data = Image.open(img_path)
        if data.mode != "RGB":
            data = data.convert("RGB")
        data = self.transforms(data)
        return data,label
 
    def __len__(self):
        return len(self.imgs)
 
    def _split_data_set(self, imags):
        """
        分类数据为训练集和验证集，根据个人数据特点设计，不通用。
        """
        val_path_list = imags[::5]
        train_path_list = []
        for item in imags:
            if item not in val_path_list:
                train_path_list.append(item)
        return train_path_list, val_path_list
 
if __name__ == "__main__":
    root = "/root/autodl-tmp/handpose_x_gesture_v1"
   
    train_dataset = hand_pose(root, train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    for data, label in train_dataloader:
        print(data.shape)
        print(label)
        break
 