'''
MilleAcariNet
'''

import torch
from torch import nn
from torch import optim

import lightning as L

class YoloV5Backbone(nn.Module):
    '''
    Pretrained YOLOV5
    '''
    def __init__(self, flavor: str='yolov5s', pretrained: bool=True):
        '''
        Initialize the model.

        Args:
          flavor (str): Variant of YOLOv5. Defaults to yolov5s.
          pretrained (bool): Use pretrained model. Defaults to True.
        '''
        super().__init__()
        self.model = torch.hub.load('ultralytics/yolov5', flavor, pretrained=pretrained)

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

    def forward(self, x):
        '''
        Model forward step calls the backbone.
        '''
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        '''Training step.'''
        x, y = batch
        yhat = self(x)
        loss = self.loss_fn(yhat, y)
        self.log('train/loss', loss)

    def validation_step(self, batch, batch_idx):
        '''Validation step.'''
        x, y = batch
        yhat = self(x)

        loss = self.loss_fn(yhat, y)
        acc = self.acc_metric(yhat, y)
        self.log('val/loss', loss, on_epoch=True)
        self.log('val/acc', acc, on_epoch=True)

    def configure_optimizers(self):
        '''Configure the optimizer to train all model parameters'''
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
