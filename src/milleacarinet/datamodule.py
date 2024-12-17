'''
Classes and functions related to data
'''

from typing import List

import os

import PIL

import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

import lightning as L

import logging

from .utils import create_logger
log = logging.getLogger(__name__)
log = create_logger(log)

class YOLODataset(Dataset):
    '''Dataset that complies with YOLO format'''
    def __init__(self, images_dir: str, labels_dir: str, image_files: List[str]=None, augment=None):
        '''
        Initializes the dataset.

        Args:
          images_dir (str): path to images
          labels_dir (str): path to labels
          image_files (List[str]): list of image files. Defaults to None (use all available image files).
          augment: TODO specify data augmentation
        '''
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        if image_files is None:
            self.image_files = [f.replace('.JPG', '.jpg') for f in os.listdir(images_dir) if f.lower().endswith('.jpg')]
        else:
            self.image_files = image_files

        self.transform = v2.Compose([
            v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
            v2.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
            v2.RandomResizedCrop(size=(224, 224), antialias=True),  # Or Resize(antialias=True) TODO
            v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # TODO
        ])

        log.info(f'Number of image files: {len(self.image_files)}')

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        label_path = os.path.join(self.labels_dir, self.image_files[idx].replace('.jpg', '.txt'))
        
        # Load image
        image = PIL.Image.open(image_path)
        image = self.transform(image)

        # Load labels
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    labels.append([class_id, x_center, y_center, width, height])
        
        # Convert to tensors
        labels = torch.tensor(labels, dtype=torch.float32)

        if self.transform:
            image, labels = self.transform(image, labels)

        return image, labels

class MilleAcariDataModule(L.LightningDataModule):
    '''
    Datamodule for the mille acari project.
    '''
    def __init__(self, images_dir, labels_dir, augment=None, f_val=0.2, num_workers=8, batch_size=32):
        '''
        Initializes the datamodule.
        
        Args:
          images_dir (str): path to images
          labels_dir (str): path to labels
          augment: TODO specify data augmentation
          f_val (float): fraction of validation data. Defaults to 0.2
          num_workers (int): number of workers in dataloader. Defaults to 8.
          batch_size (int): batch size in dataloader. Defaults to 32.
        '''
        super().__init__()
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.augment = augment
        self.f_val = f_val
        self.num_workers = num_workers
        self.batch_size = batch_size

        # will be assigned in setup
        self.train_ds = None
        self.val_ds = None

    def prepare_data(self):
        return super().prepare_data()

    def setup(self, stage):
        if stage == 'fit':
            image_files = [f.replace('.JPG', '.jpg') for f in np.sort(os.listdir(self.images_dir)) if f.lower().endswith('.jpg')]
            train_files, val_files = train_test_split(image_files, test_size=self.f_val)
            self.train_ds = YOLODataset(self.images_dir, self.labels_dir, train_files, self.augment)
            self.val_ds = YOLODataset(self.images_dir, self.labels_dir, val_files, self.augment)
        elif stage == 'test':
            raise NotImplementedError('Test stage is not implemented.')

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)