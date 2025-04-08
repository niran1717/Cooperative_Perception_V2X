# pointpillar.py
# Minimal custom PointPillar model using voxelizer and PFN encoder

import torch
import torch.nn as nn
import torch.nn.functional as F
from voxelizer import Voxelizer

class PFNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x, mask):
        x = self.linear(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = F.relu(x)
        x = x * mask.unsqueeze(-1)  # mask padded zeros
        x_max = x.max(dim=1)[0]     # [N_voxels, out_channels]
        return x_max

class PillarFeatureNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=64):
        super().__init__()
        self.pfn = PFNLayer(in_channels, out_channels)

    def forward(self, voxels, num_points):
        device = voxels.device
        mask = (voxels[:, :, 0] != 0).float()
        return self.pfn(voxels, mask)

class PointPillarsScatter(nn.Module):
    def __init__(self, output_shape):
        super().__init__()
        self.output_shape = output_shape

    def forward(self, pillar_features, coords):
        # Create empty canvas
        B, C = 1, pillar_features.shape[1]
        H, W = self.output_shape
        canvas = torch.zeros((B, C, H, W), dtype=pillar_features.dtype, device=pillar_features.device)

        for i in range(pillar_features.shape[0]):
            y, x = coords[i][1], coords[i][0]
            canvas[0, :, y, x] = pillar_features[i]
        return canvas

class PointPillarModel(nn.Module):
    def __init__(self, voxel_size, pc_range, grid_shape, num_classes = 10):
        super().__init__()
        self.voxelizer = Voxelizer(voxel_size, pc_range)
        self.feature_net = PillarFeatureNet(in_channels=4, out_channels=64)
        self.scatter = PointPillarsScatter(grid_shape[:2])

        self.backbone = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
        )

        self.cls_head = nn.Conv2d(256, num_classes, 1)  # Binary classification stub
        self.reg_head = nn.Conv2d(256, 7, 1)  # Box regression stub

    def forward(self, points_np):
        voxels_np, coords_np, num_points_np = self.voxelizer.voxelize(points_np)

        # Convert to tensors
        device = next(self.parameters()).device
        voxels = torch.tensor(voxels_np, dtype=torch.float32, device=device)
        coords = torch.tensor(coords_np, dtype=torch.long, device=device)
        num_points = torch.tensor(num_points_np, dtype=torch.int32, device=device)

        pillar_features = self.feature_net(voxels, num_points)
        spatial_features = self.scatter(pillar_features, coords)
        x = self.backbone(spatial_features)

        cls_preds = self.cls_head(x)
        reg_preds = self.reg_head(x)
        return cls_preds, reg_preds