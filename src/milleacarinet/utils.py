
import torch
import torch.nn as nn
import logging
from torchvision.transforms import v2
from torchvision import tv_tensors

import logging
thislog = logging.getLogger(__name__)

def create_logger(log, level='info'):
    '''Adapt logger to our needs.
    
    Args:
      log (Logger): Logger instance.
      
    Returns: 
      Logger.
    '''
    if level == 'info':
        log.setLevel(logging.INFO)
    elif level == 'debug':
        log.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    log.addHandler(console_handler)
    return log

thislog = create_logger(thislog)

def compute_iou(box1, box2):
    """
    Compute IoU between two boxes in xywh format.
    box1: (N, 4) [x_center, y_center, width, height]
    box2: (M, 4) [x_center, y_center, width, height]
    Returns: IoU matrix (N, M)
    """
    # Convert to x1, y1, x2, y2
    box1 = torch.cat([box1[:, :2] - box1[:, 2:] / 2,  # x1, y1
                      box1[:, :2] + box1[:, 2:] / 2], dim=-1)  # x2, y2
    box2 = torch.cat([box2[:, :2] - box2[:, 2:] / 2,  # x1, y1
                      box2[:, :2] + box2[:, 2:] / 2], dim=-1)  # x2, y2

    # Compute intersection
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
             torch.max(box1[:, None, :2], box2[:, :2])).clamp(min=0)
    inter_area = inter[:, :, 0] * inter[:, :, 1]

    # Compute union
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, None] + area2 - inter_area

    return inter_area / union  # IoU


class YoloLoss(nn.Module):
    '''Loss for object detection'''
    def __init__(self, iou_threshold=0.5, penalty_factor=0.1):
        super(YoloLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.iou_threshold = iou_threshold
        self.penalty_factor = penalty_factor

    def forward(self, batch_y_hat, batch_y, batch_obj_scores):
        """
        :param y_hat: Predicted boxes (N_pred, 4) [x_center, y_center, w, h] normalized
        :param y: Ground truth boxes (N_gt, 4) [x_center, y_center, w, h] normalized
        :param obj_scores: Objectness scores (N_pred,)
        """
        batch_size = batch_y_hat.shape[0]
        total_loc_loss = 0.0
        total_obj_loss = 0.0
        total_penalty_loss = 0.0

        for b in range(batch_size):
            y_hat = batch_y_hat[b]
            y = batch_y[b]
            obj_scores = batch_obj_scores[b]
            # Step 1: Compute IoU between predictions and ground truth
            iou_matrix = compute_iou(y_hat, y)

            # Step 2: Assign predictions to ground truth
            max_iou, match_idx = iou_matrix.max(dim=1)  # Best ground truth for each prediction
            matched_gt_mask = max_iou > self.iou_threshold  # Filter matched boxes
            unmatched_pred_mask = ~matched_gt_mask  # Unmatched predictions
        
            # Step 3: Localization Loss (for matched boxes only)
            matched_preds = y_hat[matched_gt_mask]
            matched_gt = y[match_idx[matched_gt_mask]]
            if len(matched_preds) > 0:
                loc_loss = self.mse_loss(matched_preds, matched_gt)
            else:
                loc_loss = 0.0  # No matched predictions, so no localization loss

            # Penalty term for unmatched boxes
            total_penalty_loss += self.penalty_factor * unmatched_pred_mask.sum().float()

            # Step 4: Objectness Loss
            obj_labels = torch.zeros_like(obj_scores)
            obj_labels[matched_gt_mask] = 1  # Object exists for matched predictions
            obj_loss = self.bce_loss(obj_scores, obj_labels)

            total_loc_loss += loc_loss
            total_obj_loss += obj_loss

        # Aggregate across batch
        total_loss = (total_loc_loss + total_obj_loss + total_penalty_loss) / batch_size
        return total_loss

class RandomIoUCropWithFallback(nn.Module):
    '''
    RandomIOUCrop with fallback strategy if no crop is found.
    '''
    def __init__(self, ioucrop_args, randomresizedcrop_args):
        super().__init__()

        self.ioucrop = v2.RandomIoUCrop(**ioucrop_args)
        self.rrscrop = v2.RandomResizedCrop(**randomresizedcrop_args)

    def forward(self, tv_image: tv_tensors.Image, tv_boxes: tv_tensors.BoundingBoxes):
        '''
        Apply RandomIoUCrop. If it fails, perform RandomResizedCrop.

        Args:
          tv_image: Image tv_tensor
          tv_boxes: BoundingBoxes tv_tensor 

        Returns:
          tv_image, tv_boxes: Transformed tv_tensors
        '''
        try:
            tv_image, tv_boxes = self.ioucrop(tv_image, tv_boxes)
        except ZeroDivisionError:
            thislog.info('Fallback to RandomResizedCrop')
            tv_image, tv_boxes = self.rrscrop(tv_image, tv_boxes)

        return tv_image, tv_boxes