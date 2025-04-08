# voxelizer.py
# Minimal custom voxelizer for PointPillar-style encoding

import numpy as np

class Voxelizer:
    def __init__(self, voxel_size, point_cloud_range, max_points_per_voxel=32, max_voxels=20000):
        self.voxel_size = np.array(voxel_size, dtype=np.float32)
        self.point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        self.max_points_per_voxel = max_points_per_voxel
        self.max_voxels = max_voxels

        grid_size = (self.point_cloud_range[3:] - self.point_cloud_range[:3]) / self.voxel_size
        self.grid_size = np.round(grid_size).astype(np.int32)

    def voxelize(self, points):
        points = np.asarray(points, dtype=np.float32)
        if points.ndim == 3 and points.shape[0] == 1:
            points = points[0]

        mask = np.all((points[:, :3] >= self.point_cloud_range[:3]) & (points[:, :3] < self.point_cloud_range[3:]), axis=1)
        points = points[mask]

        # Compute voxel coordinates
        voxel_coords = ((points[:, :3] - self.point_cloud_range[:3]) / self.voxel_size).astype(np.int32)
        voxel_coords = np.clip(voxel_coords, a_min=0, a_max=self.grid_size - 1)

        # Create a dict to group points by voxel index
        voxel_dict = {}
        for i in range(points.shape[0]):
            coord = tuple(voxel_coords[i])
            if coord not in voxel_dict:
                voxel_dict[coord] = []
            if len(voxel_dict[coord]) < self.max_points_per_voxel:
                voxel_dict[coord].append(points[i])

        # Sort and limit number of voxels
        voxel_coords = list(voxel_dict.keys())[:self.max_voxels]
        voxel_features = [voxel_dict[coord] for coord in voxel_coords]

        # Convert to numpy arrays
        voxels = np.zeros((len(voxel_coords), self.max_points_per_voxel, 4), dtype=np.float32)
        coords = np.zeros((len(voxel_coords), 3), dtype=np.int32)
        num_points_per_voxel = np.zeros(len(voxel_coords), dtype=np.int32)

        for i, coord in enumerate(voxel_coords):
            points_in_voxel = np.array(voxel_features[i], dtype=np.float32)
            voxels[i, :len(points_in_voxel)] = points_in_voxel
            coords[i] = coord
            num_points_per_voxel[i] = len(points_in_voxel)

        return voxels, coords, num_points_per_voxel
