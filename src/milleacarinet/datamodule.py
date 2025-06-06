'''
Classes and functions related to data
'''

from abc import ABC
from typing import List

import os

import PIL

import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torchvision.transforms import v2
from torchvision import tv_tensors

from torchvision.utils import draw_bounding_boxes
from torchvision.ops import box_convert

import lightning as L

import logging

from .utils import create_logger, collate_fn, RandomIoUCropWithFallback

log = logging.getLogger(__name__)
log = create_logger(log)

class YOLOBaseDataset(ABC):
    '''Dataset that complies with YOLO format. Base class for map-style and iterable dataset.'''
    def __init__(self, images_dir: str, labels_dir: str, image_files: List[str]=None, augment=None, size=640):
        '''
        Initializes the dataset.

        Args:
          images_dir (str): path to images
          labels_dir (str): path to labels
          image_files (List[str]): list of image files. Defaults to None (use all available image files).
          augment: TODO specify data augmentation
          size (int): Image target size. Defaults to 640.
        '''
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        if image_files is None:
            self.image_files = [f.replace('.JPG', '.jpg') for f in os.listdir(images_dir) if f.lower().endswith('.jpg')]
        else:
            self.image_files = image_files

        self.transform = v2.Compose([
            RandomIoUCropWithFallback({'min_scale': 0.5, 'trials': 100}, {'size': size}),  # Increase min_scale to avoid extreme crops
            v2.Resize((size, size)),  # Increased from 512 to 640 for better detail
            v2.RandomHorizontalFlip(0.5),
            v2.RandomVerticalFlip(0.5),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # More controlled color jitter
            v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),  # Add blur for robustness
            v2.ToDtype(torch.float32, scale=True)
        ])

        log.info(f'Number of image files: {len(self.image_files)}')

    def getitem(self, idx):
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        label_path = os.path.join(self.labels_dir, self.image_files[idx].replace('.jpg', '.txt'))

        image = PIL.Image.open(image_path)

        yolo_boxes = []
        yolo_classes = []
        with open(label_path, 'r') as f:
            line = f.readline()
            while line:
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                yolo_boxes.append([x_center, y_center, width, height])
                yolo_classes.append([class_id])
                line = f.readline()

        W, H = image.size
        log.debug(f'Image size: {W}, {H} with {len(yolo_boxes)} objects')
        log.debug(f'Original YOLO boxes: {yolo_boxes}')

        # From YOLO (0, 1) to torchvision (pixels)
        tv_boxes = torch.from_numpy(np.array(yolo_boxes).copy())
        tv_boxes[:, [0, 2]] *= W
        tv_boxes[:, [1, 3]] *= H

        log.debug(f'Torchvision boxes (pixels): {tv_boxes}')

        # Convert to tensors
        tvt_boxes = tv_tensors.BoundingBoxes(tv_boxes, canvas_size=(H, W), format='XYWH', dtype=torch.float32)
        torch_image = torch.from_numpy(np.array(image).transpose(2, 0, 1))
        tvt_image = tv_tensors.Image(torch_image)

        if self.transform:
            tvt_image, tvt_boxes = self.transform(tvt_image, tvt_boxes)
            log.debug(f'Transformed image shape: {tvt_image.shape}')
            log.debug(f'Transformed boxes before sanitization: {tvt_boxes}')
            tvt_boxes = v2.SanitizeBoundingBoxes()({'labels': tvt_boxes})['labels']
            log.debug(f'Sanitized boxes: {tvt_boxes}')

        # Back to YOLO coordinates
        tvt_boxes[:, [0, 2]] /= tvt_image.shape[-1]
        tvt_boxes[:, [1, 3]] /= tvt_image.shape[-2]

        log.debug(f'Final YOLO boxes: {tvt_boxes}')

        return tvt_image, tvt_boxes

class YOLODataset(YOLOBaseDataset, Dataset):
    def __init__(self, images_dir: str, labels_dir: str, image_files: List[str]=None, augment=None, size=512):
        '''
        Initializes the dataset.

        Args:
          images_dir (str): path to images
          labels_dir (str): path to labels
          image_files (List[str]): list of image files. Defaults to None (use all available image files).
          augment: TODO specify data augmentation
          size (int): Image target size. Defaults to 512.
        '''
        super().__init__(images_dir, labels_dir, image_files, augment, size)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        return self.getitem(idx)

class YOLOIterableDataset(YOLOBaseDataset, IterableDataset):
    def __init__(self, images_dir: str, labels_dir: str, image_files: List[str]=None, augment=None, size=512,
                 max_samples=100):
        '''
        Initializes the dataset. Iterable version.

        Args:
          images_dir (str): path to images
          labels_dir (str): path to labels
          image_files (List[str]): list of image files. Defaults to None (use all available image files).
          augment: TODO specify data augmentation
          size (int): Image target size. Defaults to 512.
          max_samples (int): Maximum number of samples generated by the iterator. Defaults to 100.
        '''
        super().__init__(images_dir, labels_dir, image_files, augment, size)

        self.max_samples = max_samples
        log.info(f'Iterable dataset, yields a maximum of {self.max_samples} samples.')

    def __len__(self):
        return self.max_samples

    def __iter__(self):
        # Generate up to max_samples
        for _ in range(self.max_samples):
            # Select a random image index
            random_ix = int(np.random.randint(0, len(self.image_files), size=1))
            # Get the image and boxes for this index
            tvt_images, tvt_boxes = self.getitem(random_ix)
            # Return the images and boxes in XYWH format (same as YOLODataset)
            yield tvt_images, tvt_boxes

class MilleAcariDataModule(L.LightningDataModule):
    '''
    Datamodule for the mille acari project.
    '''
    def __init__(self,
                 images_dir,
                 labels_dir,
                 augment=None,
                 size=512,
                 max_samples=None,
                 val_size=0.2,
                 num_workers=8,
                 batch_size=32):
        '''
        Initializes the datamodule.
        
        Args:
          images_dir (str): path to images
          labels_dir (str): path to labels
          augment: TODO specify data augmentation
          size (int): Image size.
          max_samples (int): Maximum number of samples. Defaults to None (use map-style dataset).
          val_size (float): fraction of validation data. Defaults to 0.2
          num_workers (int): number of workers in dataloader. Defaults to 8.
          batch_size (int): batch size in dataloader. Defaults to 32.
        '''
        super().__init__()
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.augment = augment
        self.size = size
        self.max_samples = max_samples
        self.val_size = val_size
        self.num_workers = num_workers
        self.batch_size = batch_size

        # will be assigned in setup
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.shuffle = False

    def prepare_data(self):
        return super().prepare_data()

    def setup(self, stage):
        if stage == 'fit':
            image_files = [f.replace('.JPG', '.jpg') for f in np.sort(os.listdir(self.images_dir)) if f.lower().endswith('.jpg')]
            train_files, val_files = train_test_split(image_files, test_size=self.val_size)

            if self.max_samples is None:
                self.train_ds = YOLODataset(self.images_dir, self.labels_dir, train_files, self.augment)
                self.val_ds = YOLODataset(self.images_dir, self.labels_dir, val_files, self.augment)
                self.shuffle = True
            else:
                self.train_ds = YOLOIterableDataset(self.images_dir, self.labels_dir, train_files, self.augment, self.size, self.max_samples)
                self.val_ds = YOLOIterableDataset(self.images_dir, self.labels_dir, val_files, self.augment, self.size, self.max_samples)

            self.test_ds = YOLODataset(self.images_dir, self.labels_dir, val_files, self.augment)

        elif stage == 'test':
            raise NotImplementedError('Test stage is not implemented.')

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=collate_fn, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=collate_fn, shuffle=self.shuffle)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=1, num_workers=self.num_workers, collate_fn=collate_fn, shuffle=self.shuffle)

    def predict_dataloader(self):
        # TODO run predictions on the unlabeled data
        return DataLoader(self.val_ds, batch_size=1, )