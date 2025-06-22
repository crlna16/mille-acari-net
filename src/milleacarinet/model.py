'''
MilleAcariNet
'''

import logging

import numpy as np

import torch
from torch import nn
from torch import optim

import lightning as L

import torch.nn as nn
import torchvision

from torchvision.utils import draw_bounding_boxes
from torchvision.ops import box_convert

from ultralytics import YOLO

from .utils import YoloLoss, create_logger

log = logging.getLogger(__name__)
log = create_logger(log)

class YoloV8Backbone(nn.Module):
    '''
    Pretrained YOLOv8 backbone
    '''
    def __init__(self,
                 model_size: str='n',  # n, s, m, l, x
                 pretrained: bool=True,
                 freeze_backbone: bool=False,
                 ):
        '''
        Initialize the YOLOv8 model.

        Args:
          model_size (str): Size of YOLOv8 model ('n', 's', 'm', 'l', 'x'). Defaults to 'n' (nano).
          pretrained (bool): Use pretrained model. Defaults to True.
          freeze_backbone (bool): Freeze backbone layers. Defaults to False.
        '''
        super().__init__()
        
        # Import here to avoid dependency issues if not using this class
        
        # Load the model - YOLOv8 with specified size
        self.model = YOLO(f"yolov8{model_size}.pt")
        
        # Extract the backbone part of the model
        self.backbone = self.model.model.model[0:9]  # Layers 0-9 contain the backbone
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Add detection head for object detection
        # This is a simplified detection head - you might want to customize this
        self.detection_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.Conv2d(256, 5, kernel_size=1)  # 4 box coords + 1 objectness score
        )
        
    def forward(self, x):
        '''Extract features using backbone and apply detection head'''
        # Get features from backbone
        features = self.backbone(x)
        
        # Apply detection head
        detections = self.detection_head(features)
        
        # Reshape to expected format [batch_size, num_predictions, 5]
        batch_size = x.shape[0]
        detections = detections.permute(0, 2, 3, 1).reshape(batch_size, -1, 5)
        
        return detections

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
            if depth > self.freeze_depth:
                val.requires_grad = True

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
                 min_obj_score: float=1e-5,
                 iou_threshold: float=0.5,
                 confidence_threshold: float=0.5,
                 ):
        '''
        Initializes MilleAcariNet.

        Args:
          backbone (nn.Module): torch backbone
          lr (float): learning rate
          min_obj_score (float): minimum object score for an object to be considered. Defaults to 1e-5.
          iou_threshold (float): IoU threshold for NMS. Defaults to 0.5.
          confidence_threshold (float): Confidence threshold for filtering predictions. Defaults to 0.5.
        '''
        super().__init__()
        self.backbone = backbone
        self.lr = lr
        self.min_obj_score = min_obj_score
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold

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

        loss = self.loss_fn(pred_boxes, y, pred_obj_scores, min_obj_score=self.min_obj_score)
        self.log('train/loss', loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        '''Validation step.'''
        x, y = batch
        yhat = self(x)

        pred_boxes = yhat[:, :, :4]
        pred_obj_scores = yhat[:, :, 4]

        loss = self.loss_fn(pred_boxes, y, pred_obj_scores, min_obj_score=self.min_obj_score)

        #acc = self.acc_metric(yhat, y)
        self.log('val/loss', loss, on_epoch=True, prog_bar=True)
        #self.log('val/acc', acc, on_epoch=True)


        # save predictions for visualization (first batch item only)
        x = x[0]
        pred_boxes = y[:, :, :4]  # boxes in XYWH format
        pred_obj_scores = torch.ones_like(pred_boxes)[:, :, 0]#y[:, :, 4]
        y = y[0]
        yhat = yhat[0]
        # Filter predictions by confidence score
        confidence_threshold = self.confidence_threshold  # Adjust this threshold as needed
        confident_mask = pred_obj_scores.squeeze(0) > confidence_threshold
        
        # Apply confidence filtering
        filtered_boxes = pred_boxes.squeeze(0)[confident_mask]
        filtered_scores = pred_obj_scores.squeeze(0)[confident_mask]
        
        log.info(f'Boxes before filtering: {pred_boxes.squeeze(0).shape[0]}')
        log.info(f'Boxes after confidence filtering: {filtered_boxes.shape[0]}')
        
        # Convert filtered boxes to XYXY format for NMS
        filtered_boxes_xyxy = box_convert(filtered_boxes, in_fmt='xywh', out_fmt='xyxy')
        
        # Apply Non-Maximum Suppression to remove overlapping boxes
        iou_threshold = self.iou_threshold  # Adjust this threshold as needed
        keep_indices = torchvision.ops.nms(
            filtered_boxes_xyxy, 
            filtered_scores,
            iou_threshold
        )
        
        # Get final predictions
        final_boxes_xyxy = filtered_boxes_xyxy[keep_indices]
        final_scores = filtered_scores[keep_indices]
        
        log.info(f'Final boxes after NMS: {final_boxes_xyxy.shape[0]}')
        
        # Visualize only the final predictions
        dbb = draw_bounding_boxes(
            x.squeeze(0), 
            final_boxes_xyxy, 
            labels=[f"{score:.2f}" for score in final_scores], 
            colors='red', 
            fill=True, 
            width=2,
            font_size=12
        )

        dbb = dbb.cpu().numpy()
        np.save(f'predictions/val_epoch_{self.current_epoch}_batch_{batch_idx}.npy', dbb)
        
    def predict_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        
        # Get raw predictions
        pred_boxes = yhat[:, :, :4]  # boxes in XYWH format
        pred_obj_scores = yhat[:, :, 4]
        
        loss = self.loss_fn(pred_boxes, y, pred_obj_scores, min_obj_score=self.min_obj_score)
        log.info(f'Predict loss: {loss:.2e}')
        
        # Filter predictions by confidence score
        confidence_threshold = self.confidence_threshold  # Adjust this threshold as needed
        confident_mask = pred_obj_scores.squeeze(0) > confidence_threshold
        
        # Apply confidence filtering
        filtered_boxes = pred_boxes.squeeze(0)[confident_mask]
        filtered_scores = pred_obj_scores.squeeze(0)[confident_mask]
        
        log.info(f'Boxes before filtering: {pred_boxes.squeeze(0).shape[0]}')
        log.info(f'Boxes after confidence filtering: {filtered_boxes.shape[0]}')
        
        # Convert filtered boxes to XYXY format for NMS
        filtered_boxes_xyxy = box_convert(filtered_boxes, in_fmt='xywh', out_fmt='xyxy')
        
        # Apply Non-Maximum Suppression to remove overlapping boxes
        iou_threshold = self.iou_threshold  # Adjust this threshold as needed
        keep_indices = torchvision.ops.nms(
            filtered_boxes_xyxy, 
            filtered_scores,
            iou_threshold
        )
        
        # Get final predictions
        final_boxes_xyxy = filtered_boxes_xyxy[keep_indices]
        final_scores = filtered_scores[keep_indices]
        
        log.info(f'Final boxes after NMS: {final_boxes_xyxy.shape[0]}')
        
        # Visualize only the final predictions
        dbb = draw_bounding_boxes(
            x.squeeze(0), 
            final_boxes_xyxy, 
            labels=[f"{score:.2f}" for score in final_scores], 
            colors='red', 
            fill=True, 
            width=2,
            font_size=12
        )
        
        return dbb.cpu().numpy()

    def configure_optimizers(self):
        '''Configure the optimizer and learning rate scheduler'''
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        
        # Add ReduceLROnPlateau scheduler that halves the learning rate when val/loss plateaus
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,    # halve the learning rate
                patience=5,    # wait 5 epochs for improvement
                verbose=True,
                min_lr=1e-6    # don't reduce lr below this value
            ),
            'monitor': 'val/loss',  # monitor validation loss
            'interval': 'epoch',
            'frequency': 1
        }
        
        return [optimizer], [scheduler]
