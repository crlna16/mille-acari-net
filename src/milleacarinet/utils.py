import torch
import torch.nn as nn
import logging
from torchvision.transforms import v2
from torchvision import tv_tensors
from torch.nn.utils.rnn import pad_sequence

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
    '''Loss for object detection with enhanced small object detection capabilities'''
    def __init__(self, iou_threshold=0.5, penalty_factor=0.1, box_loss_weight=5.0, obj_loss_weight=1.0, small_obj_scale=2.0, small_size_threshold=0.05):
        super(YoloLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')  # Changed to none for per-box weighting
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')  # Changed to none for per-instance weighting
        self.iou_threshold = iou_threshold
        self.penalty_factor = penalty_factor
        self.box_loss_weight = box_loss_weight  # Weight for localization loss
        self.obj_loss_weight = obj_loss_weight  # Weight for objectness loss
        self.small_obj_scale = small_obj_scale  # Additional weight for small objects
        self.small_size_threshold = small_size_threshold  # Threshold to define small objects (relative to image)

    def forward(self, batch_y_hat, batch_y, batch_obj_scores, min_obj_score=0, batch_class_pred=None, batch_class_target=None):
        """
        :param batch_y_hat: Predicted boxes (batch_size, N_pred, 4) [x_center, y_center, w, h] normalized
        :param batch_y: Ground truth boxes (batch_size, N_gt, 4) [x_center, y_center, w, h] normalized
        :param batch_obj_scores: Objectness scores (batch_size, N_pred)
        :param min_obj_score: minimum object score
        :param batch_class_pred: Optional class predictions (batch_size, N_pred, num_classes)
        :param batch_class_target: Optional class targets (batch_size, N_gt)
        :return: total loss
        """
        batch_size = batch_y_hat.shape[0]
        total_loc_loss = 0.0
        total_obj_loss = 0.0
        total_penalty_loss = 0.0
        total_class_loss = 0.0
        valid_batches = 0
        total_small_obj_count = 0
        total_obj_count = 0

        for b in range(batch_size):
            y_hat = batch_y_hat[b]
            y = batch_y[b]
            obj_scores = batch_obj_scores[b]

            # Filter predictions by confidence threshold
            ix = obj_scores > min_obj_score
            obj_scores = obj_scores[ix]
            y_hat = y_hat[ix]

            # Skip computation if there are no predictions after filtering
            if y_hat.shape[0] == 0:
                continue

            # Skip computation if there are no ground truth boxes
            if y.shape[0] == 0:
                # Still penalize high confidence predictions when no objects exist
                obj_loss = self.bce_loss(obj_scores, torch.zeros_like(obj_scores)).mean()
                total_obj_loss += obj_loss * self.obj_loss_weight
                valid_batches += 1
                continue

            # Identify small objects in ground truth
            # Small objects defined as having width or height < small_size_threshold
            is_small_obj = (y[:, 2] < self.small_size_threshold) | (y[:, 3] < self.small_size_threshold)
            small_obj_count = is_small_obj.sum().item()
            total_small_obj_count += small_obj_count
            total_obj_count += y.shape[0]

            # Step 1: Compute IoU between predictions and ground truth
            iou_matrix = compute_iou(y_hat, y)

            # Step 2: Assign predictions to ground truth
            max_iou, match_idx = iou_matrix.max(dim=1)  # Best ground truth for each prediction
            matched_gt_mask = max_iou > self.iou_threshold  # Filter matched boxes
            unmatched_pred_mask = ~matched_gt_mask  # Unmatched predictions
        
            # Step 3: Localization Loss (for matched boxes only) with small object emphasis
            if matched_gt_mask.sum() > 0:
                matched_preds = y_hat[matched_gt_mask]
                matched_gt_indices = match_idx[matched_gt_mask]
                matched_gt = y[matched_gt_indices]
                
                # Apply CIoU loss for better localization
                ciou_loss = 1 - bbox_ciou(
                    box_convert(matched_preds, in_fmt='xywh', out_fmt='xyxy'),
                    box_convert(matched_gt, in_fmt='xywh', out_fmt='xyxy')
                )
                
                # Scale loss for small objects
                matched_is_small = is_small_obj[matched_gt_indices]
                small_obj_weights = torch.ones_like(ciou_loss)
                small_obj_weights[matched_is_small] = self.small_obj_scale
                
                # Apply scaling and average
                loc_loss = (ciou_loss * small_obj_weights).mean()
                total_loc_loss += loc_loss * self.box_loss_weight

                # Add class loss if class predictions are provided
                if batch_class_pred is not None and batch_class_target is not None:
                    class_pred = batch_class_pred[b][ix][matched_gt_mask]
                    class_target = batch_class_target[b][matched_gt_indices]
                    class_loss = nn.functional.cross_entropy(
                        class_pred, class_target.long(), reduction='none')
                    # Scale class loss for small objects too
                    class_loss = (class_loss * small_obj_weights).mean()
                    total_class_loss += class_loss

            # Penalty term for unmatched high-confidence predictions (weighted by confidence)
            if unmatched_pred_mask.sum() > 0:
                unmatched_scores = obj_scores[unmatched_pred_mask]
                penalty = self.penalty_factor * unmatched_scores.sum()
                total_penalty_loss += penalty

            # Step 4: Objectness Loss with focus on improving recall for small objects
            obj_labels = torch.zeros_like(obj_scores)
            obj_labels[matched_gt_mask] = 1  # Object exists for matched predictions
            
            # Calculate per-instance objectness loss
            obj_loss_per_pred = self.bce_loss(obj_scores, obj_labels)
            
            # Apply higher weight to false negatives (missed objects) to improve recall
            obj_loss_weights = torch.ones_like(obj_loss_per_pred)
            # Higher penalty for false negatives (improve recall)
            fn_mask = (obj_labels == 1) & (obj_scores < 0.5)
            obj_loss_weights[fn_mask] = 2.0
            
            # Average and add to total
            obj_loss = (obj_loss_per_pred * obj_loss_weights).mean()
            total_obj_loss += obj_loss * self.obj_loss_weight
            
            valid_batches += 1

        # Prevent division by zero if all batches were skipped
        if valid_batches == 0:
            return torch.tensor(0.0, device=batch_y_hat.device, requires_grad=True)
            
        # Log statistics about small objects if any were found
        if total_obj_count > 0:
            small_obj_percentage = 100 * total_small_obj_count / total_obj_count
            if small_obj_percentage > 0:
                logging.debug(f'Small objects: {total_small_obj_count}/{total_obj_count} ({small_obj_percentage:.1f}%)')
            
        # Normalize by the number of valid batches
        total_loss = (total_loc_loss + total_obj_loss + total_penalty_loss + total_class_loss) / valid_batches
        return total_loss

def bbox_ciou(box1, box2, eps=1e-7):
    """
    Calculate CIoU (Complete IoU) between two sets of bounding boxes
    :param box1: First set of boxes (N, 4) in xyxy format
    :param box2: Second set of boxes (N, 4) in xyxy format
    :return: CIoU loss (1 - CIoU)
    """
    # Get bounding box coordinates
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.unbind(1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.unbind(1)
    
    # Calculate area of boxes
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    # Calculate intersection area
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # Calculate union area
    union_area = b1_area + b2_area - inter_area + eps
    
    # Calculate IoU
    iou = inter_area / union_area
    
    # Get enclosing box
    enclose_x1 = torch.min(b1_x1, b2_x1)
    enclose_y1 = torch.min(b1_y1, b2_y1)
    enclose_x2 = torch.max(b1_x2, b2_x2)
    enclose_y2 = torch.max(b1_y2, b2_y2)
    
    # Calculate diagonal distance squared
    c2 = torch.pow(enclose_x2 - enclose_x1, 2) + torch.pow(enclose_y2 - enclose_y1, 2) + eps
    
    # Calculate centers of boxes
    b1_cx = (b1_x1 + b1_x2) / 2
    b1_cy = (b1_y1 + b1_y2) / 2
    b2_cx = (b2_x1 + b2_x2) / 2
    b2_cy = (b2_y1 + b2_y2) / 2
    
    # Calculate central distance squared
    center_dist2 = torch.pow(b1_cx - b2_cx, 2) + torch.pow(b1_cy - b2_cy, 2)
    
    # Calculate v and alpha for aspect ratio consistency
    w1 = b1_x2 - b1_x1
    h1 = b1_y2 - b1_y1
    w2 = b2_x2 - b2_x1
    h2 = b2_y2 - b2_y1
    
    v = (4 / (torch.pi ** 2)) * torch.pow(torch.atan(w1 / (h1 + eps)) - torch.atan(w2 / (h2 + eps)), 2)
    alpha = v / (1 - iou + v + eps)
    
    # CIoU
    ciou = iou - (center_dist2 / c2 + alpha * v)
    
    return ciou

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
            thislog.debug('Fallback to RandomResizedCrop')
            tv_image, tv_boxes = self.rrscrop(tv_image, tv_boxes)

        return tv_image, tv_boxes

def collate_fn(batch):
    """
    Custom collate function for object detection datasets.

    This is to collate samples with different number of bounding boxes.

    Args:
        batch: A list of samples, where each sample is a tuple:
               (image, bounding_boxes), where
               - image: torch.Tensor of shape (3, H, W)
               - bounding_boxes: torch.Tensor of shape (N, 4)
    """
    images = []
    bboxes = []

    for image, boxes in batch:
        if boxes.size(0) > 0:  # Only add non-empty bounding boxes
            images.append(image)
            bboxes.append(boxes)

    if len(images) == 0 or len(bboxes) == 0:
        raise ValueError("All bounding boxes are empty in the batch.")

    # Stack images into a single tensor of shape (batch_size, 3, H, W)
    images = torch.stack(images, dim=0)

    # Pad bounding boxes to the same size
    bboxes = pad_sequence(bboxes, batch_first=True, padding_value=0)

    return images, bboxes
