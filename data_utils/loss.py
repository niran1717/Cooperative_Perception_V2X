import torch
import torch.nn as nn

class PointPillarLoss(nn.Module):
    """
    A simpler LiDAR-only detection loss:
      - Classification: CrossEntropyLoss
      - Regression: SmoothL1Loss
    """

    def __init__(self, num_classes=3):
        super(PointPillarLoss, self).__init__()
        self.num_classes = num_classes
        self.class_loss_fn = nn.CrossEntropyLoss()
        self.reg_loss_fn = nn.SmoothL1Loss()

    def forward(self, cls_preds, reg_preds, cls_targets, reg_targets):
        """
        cls_preds:  (B, C, H, W) - classification logits
        reg_preds:  (B, 7, H, W) - 3D box predictions
        cls_targets:(B, H, W)    - ground truth class indices
        reg_targets:(B, 7, H, W) - ground truth box params
        """

        # 1) Classification
        # Permute from (B, C, H, W) → (B, H, W, C), then flatten
        B, C, H, W = cls_preds.shape
        cls_preds_flat = cls_preds.permute(0, 2, 3, 1).reshape(-1, C)
        cls_targets_flat = cls_targets.reshape(-1)
        class_loss = self.class_loss_fn(cls_preds_flat, cls_targets_flat)

        # 2) Bounding box regression
        # Permute from (B, 7, H, W) → (B, H, W, 7), then flatten
        B, D, H, W = reg_preds.shape  # D=7 for x,y,z,w,l,h,yaw

        
        reg_preds_flat = reg_preds.permute(0, 2, 3, 1).reshape(-1, D)
        reg_targets_flat = reg_targets.permute(0, 2, 3, 1).reshape(-1, D)
        reg_loss = self.reg_loss_fn(reg_preds_flat, reg_targets_flat)

        # Combine
        total_loss = class_loss + reg_loss
        return total_loss
