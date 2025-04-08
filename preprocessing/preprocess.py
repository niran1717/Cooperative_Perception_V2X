# preprocess_tumtraf.py (Final Version - Recursive PCD Search)

import os
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from pypcd import pypcd
import json

def convert_pcd_to_bin(pcd_path, bin_path):
    pc = pypcd.PointCloud.from_path(pcd_path)
    points = np.zeros([pc.pc_data.shape[0], 4], dtype=np.float32)
    points[:, 0] = pc.pc_data['x']
    points[:, 1] = pc.pc_data['y']
    points[:, 2] = pc.pc_data['z']
    points[:, 3] = pc.pc_data['intensity']
    points.tofile(bin_path)

def gather_info_from_jsons(split_dir, split_name):
    json_dir = split_dir / "labels_point_clouds" / "s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered"
    point_cloud_root = split_dir / "point_clouds"
    lidar_bin_dir = split_dir / "lidar_bin"
    lidar_bin_dir.mkdir(parents=True, exist_ok=True)

    infos = []
    json_files = sorted(json_dir.glob("*.json"))

    for json_path in tqdm(json_files, desc=f"Processing {split_name}"):
        with open(json_path, 'r') as f:
            data = json.load(f)

        frames = data['openlabel']['frames']
        frame_key = next(iter(frames))  # handle dynamic keys
        frame_info = frames[frame_key]
        frame_props = frame_info['frame_properties']

        images = frame_props.get("image_file_names", [])
        pcds = frame_props.get("point_cloud_file_names", [])

        for pcd_name in pcds:
            # 🔁 Recursive search for PCD file
            pcd_path = next(point_cloud_root.rglob(pcd_name), None)
            if pcd_path is None or not pcd_path.exists():
                print(f"[Warning] Missing PCD: {pcd_name}")
                continue

            bin_path = lidar_bin_dir / (pcd_path.stem + ".bin")
            convert_pcd_to_bin(pcd_path, bin_path)

            info = {
                'lidar_path': str(bin_path),
                'frame_id': pcd_path.stem,
                'metadata': {
                    'json_path': str(json_path),
                    'image_file_names': images,
                    'point_cloud_file_names': pcds
                },
            }
            infos.append(info)

    # Save infos.pkl
    info_pkl = split_dir / f"infos_{split_name}.pkl"
    with open(info_pkl, 'wb') as f:
        pickle.dump(infos, f)

    print(f"Saved {len(infos)} samples to {info_pkl}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Path to dataset root")
    args = parser.parse_args()

    root = Path(args.root)
    for split in ["train", "val", "test"]:
        split_dir = root / split
        if split_dir.exists():
            gather_info_from_jsons(split_dir, split)
        else:
            print(f"[Warning] Split not found: {split_dir}")

if __name__ == '__main__':
    main()
