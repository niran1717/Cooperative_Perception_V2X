# tumtraf_dataset.py
# Custom PyTorch Dataset for TUMTraf-V2X

import os
import pickle
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import json
from label_parser import parse_labels_from_json
from torchvision import transforms



class TUMTrafDataset(Dataset):
    def __init__(self, info_path, modality='lidar', camera_root=None, transform=None):
        """
        Args:
            info_path (str): path to infos_*.pkl file
            modality (str): 'lidar', 'camera', or 'fusion'
            camera_root (str): root folder containing all 5 camera folders
            transform (callable, optional): transformations on data
        """
        super().__init__()
        self.info_path = info_path
        self.modality = modality
        self.camera_root = camera_root
        self.transform = transform
        self.image_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # or any fixed resolution
        transforms.ToTensor()
])


        with open(info_path, 'rb') as f:
            self.infos = pickle.load(f)

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, idx):
        info = self.infos[idx]
        sample = {'frame_id': info['frame_id']}

        if self.modality in ['lidar', 'fusion']:
            lidar_path = info['lidar_path']
            lidar_points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
            sample['lidar'] = lidar_points

        if self.modality in ['camera', 'fusion']:
            image_paths = info['metadata'].get('image_file_names', [])
            image_dict = {}
            for img_name in image_paths:
                # determine subfolder from image name
                if 's110_camera_basler_east_8mm' in img_name:
                    sub = 's110_camera_basler_east_8mm'
                elif 's110_camera_basler_north_8mm' in img_name:
                    sub = 's110_camera_basler_north_8mm'
                elif 's110_camera_basler_south1_8mm' in img_name:
                    sub = 's110_camera_basler_south1_8mm'
                elif 's110_camera_basler_south2_8mm' in img_name:
                    sub = 's110_camera_basler_south2_8mm'
                elif 'vehicle_camera_basler_16mm' in img_name:
                    sub = 'vehicle_camera_basler_16mm'
                else:
                    continue

                img_path = os.path.join(self.camera_root, sub, img_name)
                if os.path.exists(img_path):
                    image = Image.open(img_path).convert('RGB')
                    image = self.image_transform(image)

                    image_dict[sub] = np.array(image)
                else:
                    image_dict[sub] = None

            sample['camera'] = image_dict

                # ✅ NEW: Parse labels from json
        
        json_path = info['metadata']['json_path']
        labels = parse_labels_from_json(json_path)
        sample['labels'] = labels
        sample['metadata'] = info['metadata']

        if self.transform:
            sample = self.transform(sample)

        return sample
