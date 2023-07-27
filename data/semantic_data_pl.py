
import os
from typing import Any
import cv2 as cv
import PIL
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
from utils import seg_transform
import data.semantic_data as sd
from config import *

class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self,train_data_dirs,val_data_dirs) :
        super().__init__()
        self.save_hyperparameters()
        self.train_data_dirs = train_data_dirs
        self.val_data_dirs = val_data_dirs
        self.train_transforms = seg_transform.Compose(
                [seg_transform.RandBlur(),
                 seg_transform.ColorJitter(),
                 seg_transform.RandomCrop(),
                 seg_transform.RandNoise(),
                 seg_transform.RandHSV(),
                 seg_transform.RandomErasing(),
                 seg_transform.LRFlip(),
                 seg_transform.RandPerspective(),
                 seg_transform.Resize(416,224),
                 seg_transform.ArrayToTensor()])

        self.val_transforms = seg_transform.Compose([
                 seg_transform.Resize(416,224),
                 seg_transform.ArrayToTensor()])
        
    def prepare_data(self):
        pass 
    
    def setup(self, stage=None) -> None:
        self.train_datasets = sd.SemanticDataset(self.train_data_dirs,self.train_transforms)
        self.val_datasets = sd.SemanticDataset(self.train_data_dirs,self.val_transforms)
        
    def train_dataloader(self):
        return DataLoader(self.train_datasets,
                   batch_size=BATCH_SIZE,
                   shuffle=False,
                   drop_last=True,
                   num_workers=16)
     
    def val_dataloader(self) :
        return DataLoader(self.val_datasets,
                   batch_size=BATCH_SIZE,
                   shuffle=False,
                   drop_last=True,
                   num_workers=16)
    


