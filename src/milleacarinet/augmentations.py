class SmallObjectAugmentation(nn.Module):
    """
    A collection of augmentation techniques specifically designed for small object detection.
    This class implements multiple specialized augmentations that preserve small objects
    and can be configured based on the specific needs of the dataset.
    """
    def __init__(self, 
                 mosaic_prob=0.3, 
                 cutout_prob=0.3,
                 mixup_prob=0.15,
                 copy_paste_prob=0.3,
                 small_obj_threshold=0.05):
        """
        Initialize the small object augmentation pipeline.
        
        Args:
            mosaic_prob: Probability of applying mosaic augmentation
            cutout_prob: Probability of applying cutout augmentation
            mixup_prob: Probability of applying mixup augmentation
            copy_paste_prob: Probability of applying copy-paste augmentation
            small_obj_threshold: Size threshold to define small objects (relative to image)
        """
        super().__init__()
        self.mosaic_prob = mosaic_prob
        self.cutout_prob = cutout_prob
        self.mixup_prob = mixup_prob
        self.copy_paste_prob = copy_paste_prob
        self.small_obj_threshold = small_obj_threshold
        
    def apply_mosaic(self, images, boxes_list):
        """
        Apply mosaic augmentation by combining 4 images into one.
        This increases the context and number of small objects in a single image.
        """
        return mosaic_augmentation(images, boxes_list)
        
    def apply_copy_paste(self, image, boxes):
        """
        Copy small objects and paste them elsewhere in the same image.
        This increases the number of small objects for training.
        """
        return copy_paste_small_objects(image, boxes, small_obj_threshold=self.small_obj_threshold)
        
    def apply_cutout(self, image, boxes):
        """
        Apply cutout augmentation that avoids small objects.
        This improves robustness without removing valuable small object pixels.
        """
        # Implementation of cutout that avoids small objects
        # ...
        
    def apply_mixup(self, image1, boxes1, image2, boxes2):
        """
        Apply mixup augmentation with attention to small objects.
        """
        # Implementation of mixup augmentation
        # ...
        
    def forward(self, image, boxes):
        """
        Apply the augmentation pipeline with probabilities.
        """
        # Choose augmentations based on probabilities
        # ...
        
        return augmented_image, augmented_boxes

class MultiScaleTransform(nn.Module):
    """
    Multi-scale training augmentation that randomly changes the image resolution
    during training while preserving small objects.
    """
    def __init__(self, min_size=320, max_size=800, target_size=640, scale_steps=32):
        """
        Initialize multi-scale transform.
        
        Args:
            min_size: Minimum image size for training
            max_size: Maximum image size for training
            target_size: Default target size when not randomly scaled
            scale_steps: Step size between scales
        """
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.target_size = target_size
        self.scale_steps = scale_steps
        self.scales = list(range(min_size, max_size + scale_steps, scale_steps))
        
    def forward(self, image, boxes):
        """
        Apply multi-scale transformation with random size selection.
        """
        # Randomly select size with higher probability for larger sizes (better for small objects)
        if torch.rand(1).item() < 0.8:  # 80% of the time use random scale
            size_idx = torch.randint(0, len(self.scales), (1,)).item()
            new_size = self.scales[size_idx]
        else:
            new_size = self.target_size
            
        # Resize image while maintaining aspect ratio
        h, w = image.shape[-2:]
        scale = new_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Apply resize transform
        resized_image = nn.functional.interpolate(
            image.unsqueeze(0), size=(new_h, new_w), mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        # Update box coordinates based on scale
        scaled_boxes = boxes.clone()
        scaled_boxes[:, [0, 2]] *= (new_w / w)  # scale x coordinates
        scaled_boxes[:, [1, 3]] *= (new_h / h)  # scale y coordinates
        
        return resized_image, scaled_boxes

def copy_paste_small_objects(image, boxes, labels=None, small_obj_threshold=0.05, 
                           max_copies=3, min_distance=0.1):
    """
    Copy small objects and paste them elsewhere in the same image.
    
    Args:
        image: Tensor image of shape [C, H, W]
        boxes: Bounding boxes in [x_center, y_center, width, height] format
        labels: Optional class labels for the boxes
        small_obj_threshold: Size threshold to define a small object
        max_copies: Maximum number of copies to make per small object
        min_distance: Minimum distance between pasted objects (as fraction of image size)
        
    Returns:
        Augmented image and boxes (and optionally labels)
    """
    if boxes.shape[0] == 0:
        if labels is not None:
            return image, boxes, labels
        return image, boxes
    
    # Identify small objects
    is_small = (boxes[:, 2] < small_obj_threshold) | (boxes[:, 3] < small_obj_threshold)
    small_idx = torch.where(is_small)[0]
    
    if len(small_idx) == 0:
        if labels is not None:
            return image, boxes, labels
        return image, boxes
    
    C, H, W = image.shape
    new_boxes = boxes.clone()
    if labels is not None:
        new_labels = labels.clone()
    
    # For each small object, make copies
    for idx in small_idx:
        # Get the small object details
        box = boxes[idx].clone()
        if labels is not None:
            label = labels[idx].clone()
        
        # Number of copies for this object (random between 1 and max_copies)
        n_copies = torch.randint(1, max_copies + 1, (1,)).item()
        
        # Convert from center format to pixel coordinates
        x_center, y_center = box[0], box[1]
        width, height = box[2], box[3]
        
        # Calculate box in pixel coordinates
        x1 = int((x_center - width/2) * W)
        y1 = int((y_center - height/2) * H)
        x2 = int((x_center + width/2) * W)
        y2 = int((y_center + height/2) * H)
        
        # Extract the object patch
        object_patch = image[:, y1:y2, x1:x2].clone()
        
        # Create each copy
        for _ in range(n_copies):
            # Find a valid new position (avoid overlapping with existing boxes)
            attempts = 0
            valid_position = False
            
            while not valid_position and attempts < 10:
                # Choose a random position for the new center
                new_x_center = torch.rand(1).item()
                new_y_center = torch.rand(1).item()
                
                # Check minimum distance from other boxes
                valid_position = True
                for b in new_boxes:
                    dist_x = abs(new_x_center - b[0])
                    dist_y = abs(new_y_center - b[1])
                    if dist_x < (width/2 + b[2]/2 + min_distance) and \
                       dist_y < (height/2 + b[3]/2 + min_distance):
                        valid_position = False
                        break
                
                attempts += 1
            
            if not valid_position:
                continue
                
            # Calculate new box in pixel coordinates
            new_x1 = int((new_x_center - width/2) * W)
            new_y1 = int((new_y_center - height/2) * H)
            new_x2 = int((new_x_center + width/2) * W)
            new_y2 = int((new_y_center + height/2) * H)
            
            # Ensure the coordinates are within image boundaries
            new_x1 = max(0, min(W-1, new_x1))
            new_y1 = max(0, min(H-1, new_y1))
            new_x2 = max(0, min(W-1, new_x2))
            new_y2 = max(0, min(H-1, new_y2))
            
            # Skip if the box is too small after boundary adjustment
            if new_x2 - new_x1 < 3 or new_y2 - new_y1 < 3:
                continue
                
            # Get actual dimensions after boundary checks
            actual_width = (new_x2 - new_x1) / W
            actual_height = (new_y2 - new_y1) / H
            actual_x_center = (new_x1 + (new_x2 - new_x1) / 2) / W
            actual_y_center = (new_y1 + (new_y2 - new_y1) / 2) / H
            
            # Paste the object at the new position
            # Resize if necessary to match the target area
            if object_patch.shape[1:] != (new_y2 - new_y1, new_x2 - new_x1):
                resized_patch = torch.nn.functional.interpolate(
                    object_patch.unsqueeze(0),
                    size=(new_y2 - new_y1, new_x2 - new_x1),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            else:
                resized_patch = object_patch
                
            # Paste the object
            image[:, new_y1:new_y2, new_x1:new_x2] = resized_patch
            
            # Add the new box
            new_box = torch.tensor([actual_x_center, actual_y_center, 
                                   actual_width, actual_height])
            new_boxes = torch.cat([new_boxes, new_box.unsqueeze(0)], dim=0)
            
            # Add the new label if needed
            if labels is not None:
                new_labels = torch.cat([new_labels, label.unsqueeze(0)], dim=0)
    
    if labels is not None:
        return image, new_boxes, new_labels
    return image, new_boxes

def mosaic_augmentation(images, boxes_list, labels_list=None, img_size=640):
    """
    Implement mosaic augmentation which combines 4 images into one.
    This is especially effective for small object detection as it:
    1. Increases the context around small objects
    2. Creates more diverse scenes with multiple small objects
    3. Helps learn from limited training data
    
    Args:
        images: List of 4 tensor images each with shape [C, H, W]
        boxes_list: List of 4 bounding box tensors in XYWH format
        labels_list: Optional list of 4 label tensors
        img_size: Output mosaic image size
        
    Returns:
        A mosaic image with combined objects and adjusted bounding boxes
    """
    assert len(images) == 4, "Mosaic requires exactly 4 images"
    
    # Initialize mosaic image
    mosaic_img = torch.zeros((3, img_size, img_size), dtype=images[0].dtype)
    
    # Center point of the mosaic
    cx = int(random.uniform(img_size // 4, 3 * img_size // 4))
    cy = int(random.uniform(img_size // 4, 3 * img_size // 4))
    
    # Combine bounding boxes and labels from all images
    combined_boxes = []
    if labels_list is not None:
        combined_labels = []
    
    # Process each of the four images
    for i, (img, boxes) in enumerate(zip(images, boxes_list)):
        # Get original image height and width
        c, h, w = img.shape
        
        # Calculate placement coordinates
        if i == 0:  # top-left
            x1a, y1a, x2a, y2a = 0, 0, cx, cy
            x1b, y1b, x2b, y2b = w - cx, h - cy, w, h
        elif i == 1:  # top-right
            x1a, y1a, x2a, y2a = cx, 0, img_size, cy
            x1b, y1b, x2b, y2b = 0, h - cy, cx, h
        elif i == 2:  # bottom-left
            x1a, y1a, x2a, y2a = 0, cy, cx, img_size
            x1b, y1b, x2b, y2b = w - cx, 0, w, cy
        elif i == 3:  # bottom-right
            x1a, y1a, x2a, y2a = cx, cy, img_size, img_size
            x1b, y1b, x2b, y2b = 0, 0, cx, cy
        
        # Place the image in the mosaic
        mosaic_img[:, y1a:y2a, x1a:x2a] = img[:, y1b:y2b, x1b:x2b]
        
        # Adjust the bounding box coordinates if there are any
        if len(boxes) > 0:
            # Convert from normalized to pixel coordinates
            boxes_pixels = boxes.clone()
            boxes_pixels[:, 0] = boxes[:, 0] * w
            boxes_pixels[:, 1] = boxes[:, 1] * h
            boxes_pixels[:, 2] = boxes[:, 2] * w
            boxes_pixels[:, 3] = boxes[:, 3] * h
            
            # Adjust coordinates to the new position in mosaic
            # Create new boxes in absolute pixel format (x_center, y_center, width, height)
            boxes_pixels[:, 0] = boxes_pixels[:, 0] - x1b + x1a
            boxes_pixels[:, 1] = boxes_pixels[:, 1] - y1b + y1a
            
            # Filter out boxes that are outside of the mosaic
            valid_boxes = (
                (boxes_pixels[:, 0] - boxes_pixels[:, 2] / 2 > 0) &
                (boxes_pixels[:, 1] - boxes_pixels[:, 3] / 2 > 0) &
                (boxes_pixels[:, 0] + boxes_pixels[:, 2] / 2 < img_size) &
                (boxes_pixels[:, 1] + boxes_pixels[:, 3] / 2 < img_size)
            )
            
            # Keep only valid boxes
            if valid_boxes.any():
                valid_boxes_pixels = boxes_pixels[valid_boxes]
                
                # Convert back to normalized coordinates
                valid_boxes_norm = valid_boxes_pixels.clone()
                valid_boxes_norm[:, 0] = valid_boxes_pixels[:, 0] / img_size
                valid_boxes_norm[:, 1] = valid_boxes_pixels[:, 1] / img_size
                valid_boxes_norm[:, 2] = valid_boxes_pixels[:, 2] / img_size
                valid_boxes_norm[:, 3] = valid_boxes_pixels[:, 3] / img_size
                
                combined_boxes.append(valid_boxes_norm)
                
                # Also update labels if provided
                if labels_list is not None:
                    combined_labels.append(labels_list[i][valid_boxes])
    
    # Combine all valid boxes
    if combined_boxes:
        final_boxes = torch.cat(combined_boxes, dim=0)
        
        if labels_list is not None:
            if combined_labels:
                final_labels = torch.cat(combined_labels, dim=0)
                return mosaic_img, final_boxes, final_labels
            return mosaic_img, final_boxes, torch.tensor([])
        
        return mosaic_img, final_boxes
    
    # Return empty tensors if no valid boxes
    if labels_list is not None:
        return mosaic_img, torch.zeros((0, 4)), torch.tensor([])
    return mosaic_img, torch.zeros((0, 4))