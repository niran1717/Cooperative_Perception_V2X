import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class LiDARBackbone(nn.Module):
    """Feature extractor for LiDAR point clouds using PointPillars."""
    def __init__(self):
        super(LiDARBackbone, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x

class CameraBackbone(nn.Module):
    """Feature extractor for images using a ResNet-50 backbone."""
    def __init__(self):
        super(CameraBackbone, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  # Remove FC layer
    
    def forward(self, x):
        return self.feature_extractor(x)

class FusionModule(nn.Module):
    """Fuses LiDAR and Camera features."""
    def __init__(self, lidar_channels=256, cam_channels=2048, fused_channels=512):
        super(FusionModule, self).__init__()
        self.lidar_fc = nn.Linear(lidar_channels, fused_channels)
        self.cam_fc = nn.Linear(cam_channels, fused_channels)
        self.fusion_layer = nn.Linear(fused_channels * 2, fused_channels)

    def forward(self, lidar_feat, cam_feat):
        lidar_feat = F.relu(self.lidar_fc(lidar_feat.mean(dim=[2, 3])))  # Global avg pooling
        cam_feat = F.relu(self.cam_fc(cam_feat.mean(dim=[2, 3])))
        fused_feat = torch.cat([lidar_feat, cam_feat], dim=1)
        return F.relu(self.fusion_layer(fused_feat))

class DetectionHead(nn.Module):
    """Predicts 3D bounding boxes from fused features."""
    def __init__(self, fused_channels=512, num_classes=3):
        super(DetectionHead, self).__init__()
        self.fc1 = nn.Linear(fused_channels, 256)
        self.fc2 = nn.Linear(256, num_classes * 7)  # 7 values per box: (x, y, z, w, h, d, θ)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class CoopDet3D(nn.Module):
    """Cooperative 3D Object Detection Model."""
    def __init__(self):
        super(CoopDet3D, self).__init__()
        self.lidar_backbone = LiDARBackbone()
        self.camera_backbone = CameraBackbone()
        self.fusion_module = FusionModule()
        self.detection_head = DetectionHead()

    def forward(self, lidar_input, cam_input):
        lidar_feat = self.lidar_backbone(lidar_input)
        cam_feat = self.camera_backbone(cam_input)
        fused_feat = self.fusion_module(lidar_feat, cam_feat)
        predictions = self.detection_head(fused_feat)
        return predictions
