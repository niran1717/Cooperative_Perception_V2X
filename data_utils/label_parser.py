# label_parser.py (dynamic class mapping)

import json
import numpy as np
import math

# Keeps global class-to-index mapping
class ClassMap:
    def __init__(self):
        self.class2id = {}
        self.id2class = []

    def get_class_id(self, name):
        name = name.lower()
        if name not in self.class2id:
            self.class2id[name] = len(self.id2class)
            self.id2class.append(name)
        return self.class2id[name]

CLASS_MAP = ClassMap()

def parse_labels_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    frame_key = next(iter(data['openlabel']['frames']))
    objects = data['openlabel']['frames'][frame_key].get("objects", {})

    labels = []
    for obj_id, obj in objects.items():
        obj_data = obj.get("object_data", {})
        obj_type = obj_data.get("type", "unknown").lower()
        cuboid = obj_data.get("cuboid", {}).get("val", None)

        if cuboid is None:
            continue

        x, y, z = cuboid[0:3]
        sin_yaw, cos_yaw = cuboid[5], cuboid[6]
        l, w, h = cuboid[7], cuboid[8], cuboid[9]

        yaw = math.atan2(sin_yaw, cos_yaw)
        class_id = CLASS_MAP.get_class_id(obj_type)

        labels.append({
            'bbox': [x, y, z, w, l, h, yaw],
            'class': class_id,
            'class_name': obj_type
        })

    return labels
