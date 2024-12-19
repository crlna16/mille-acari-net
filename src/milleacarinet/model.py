'''
MilleAcariNet
'''

import torch
from torch import nn
from torch import optim

import lightning as L

import torch.nn as nn

from .utils import YoloLoss

class YoloV5Backbone(nn.Module):
    '''
    Pretrained YOLOV5
    '''
    def __init__(self,
                 flavor: str='yolov5s',
                 pretrained: bool=True,
                 freeze_depth: int=0,
                 ):
        '''
        Initialize the model.

        Args:
          flavor (str): Variant of YOLOv5. Defaults to yolov5s.
          pretrained (bool): Use pretrained model. Defaults to True.
          freeze_depth (int): Freeze layers until this depth. Defaults to 0.
        '''
        super().__init__()
        self.model = torch.hub.load('ultralytics/yolov5', flavor, pretrained=pretrained)
        self.freeze_depth = freeze_depth

        for key, val in self.model.named_parameters():
            depth = int(key.split('.')[3])
            print(depth, key, val.requires_grad)
            if depth > self.freeze_depth:
                val.requires_grad = True
            print(depth, key, val.requires_grad)

    def forward(self, x):
        '''Apply model'''
        return self.model(x)


class MilleAcariNet(L.LightningModule):
    '''
    Model for the mille acari project.

    Instance segmentation with bounding boxes, flexible backbone.
    '''
    def __init__(self,
                 backbone: nn.Module,
                 lr: float,
                 ):
        '''
        Initializes MilleAcariNet.

        Args:
          backbone (nn.Module): torch backbone
          lr (float): learning rate
        '''
        super().__init__()
        self.backbone = backbone
        self.lr = lr

        self.loss_fn = YoloLoss()
        self.acc_metric = None # TODO

    def forward(self, x):
        '''
        Model forward step calls the backbone.
        '''
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        '''Training step.'''
        x, y = batch
        yhat = self(x)
        pred_boxes = yhat[:, :, :4]
        pred_obj_scores = yhat[:, :, 4]

        loss = self.loss_fn(pred_boxes, y, pred_obj_scores)
        self.log('train/loss', loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        '''Validation step.'''
        x, y = batch
        yhat = self(x)

        pred_boxes = yhat[:, :, :4]
        pred_obj_scores = yhat[:, :, 4]

        loss = self.loss_fn(pred_boxes, y, pred_obj_scores)
        #acc = self.acc_metric(yhat, y)
        self.log('val/loss', loss, on_epoch=True, prog_bar=True)
        #self.log('val/acc', acc, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)

        pred_boxes = yhat[:, :, :4]
        pred_obj_scores = yhat[:, :, 4]

        loss = self.loss_fn(pred_boxes, y, pred_obj_scores)
        return loss

    def configure_optimizers(self):
        '''Configure the optimizer to train all model parameters'''
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
