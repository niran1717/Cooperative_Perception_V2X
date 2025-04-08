import numpy as np
import torch

class TargetAssigner:
    def __init__(self, grid_size, pc_range, num_classes):
        self.grid_size = grid_size  # [H, W]
        self.pc_range = pc_range    # [x_min, y_min, z_min, x_max, y_max, z_max]
        self.num_classes = num_classes

    def assign(self, gt_boxes, gt_classes):
        """
        gt_boxes: List of [x, y, z, w, l, h, yaw] 3D boxes
        gt_classes: List of class indices (int), same length as gt_boxes

        Returns:
        - cls_target: (H, W) class map
        - reg_target: (7, H, W) box regression map
        """
        H, W = self.grid_size
        cls_target = torch.zeros((H, W), dtype=torch.long)
        reg_target = torch.zeros((7, H, W), dtype=torch.float32)

        for box, cls in zip(gt_boxes, gt_classes):
            x, y, z, w, l, h, yaw = box

            # Map x, y to grid cell
            x_min, y_min, _, x_max, y_max, _ = self.pc_range
            dx = (x_max - x_min) / W
            dy = (y_max - y_min) / H

            col = int((x - x_min) / dx)
            row = int((y - y_min) / dy)

            if 0 <= row < H and 0 <= col < W:
                cls_target[row, col] = cls
                reg_target[:, row, col] = torch.tensor([x, y, z, w, l, h, yaw])

        return cls_target, reg_target
