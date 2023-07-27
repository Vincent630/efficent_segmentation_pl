import os
from typing import Any
import cv2 as cv
import PIL
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
# print(os.getcwd())

from utils import seg_transform



class SemanticDataset(Dataset):
    # return img:[bhwc],mask:[bhw]
    def __init__(self,data_dir,transforms):
        super().__init__()
        self.img_dir = None
        self.mask_dir = None
        self.data_list = list()
        self.data_dir = data_dir
        self.train_transforms = transforms
        # seg_transform.Compose(
        #     [seg_transform.RandBlur(),
        #          seg_transform.ColorJitter(),
        #          seg_transform.RandomCrop(),
        #          seg_transform.RandNoise(),
        #          seg_transform.RandHSV(),
        #          seg_transform.RandomErasing(),
        #          seg_transform.LRFlip(),
        #          seg_transform.RandPerspective(),
        #          seg_transform.Resize(448,256)]
        #     )
        if isinstance( self.data_dir ,str):
            self.data_dir = [self.data_dir]
        if isinstance(self.data_dir,list):
            self.data_dir = self.data_dir
        for folder_dir in self.data_dir :
            img_dir = os.path.join(folder_dir,"img")
            mask_dir = os.path.join(folder_dir,"mask_2cls_material")
            img_names = set([item[:-4] for item in os.listdir(img_dir)])
            mask_names = set([item[:-4] for item in os.listdir(mask_dir)])
            names = img_names.intersection(mask_names)
            for name in names:
                img_path = os.path.join(img_dir,"{:s}.jpg".format(name))
                mask_path = os.path.join(mask_dir,"{:s}.png".format(name))
                if os.path.exists(img_path) and os.path.exists(mask_path):
                    # self.data_list.append((img_path,mask_path))
                    self.data_list.append((img_path,mask_path,name))


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img_path = self.data_list[index][0]
        mask_path = self.data_list[index][1]
        data_name = self.data_list[index][2]
        
        image = cv.imread(img_path)
        mask = cv.imread(mask_path,cv.IMREAD_GRAYSCALE)
        if self.train_transforms is not None:
            image,mask = self.train_transforms(image,mask)
           
    
    
        return image,mask,data_name
        
if "__main__" == __name__:
    root_dirs = [r"D:\1\camera_ai\test"]
    dataset = SemanticDataset(root_dirs)
    datas = DataLoader(dataset,batch_size=4,shuffle=False,drop_last=True)
    for im ,ms ,nm in datas:
        print("after transform ",im.shape,ms.shape,nm)
        
        # print(im)
    
        

        