# coding=utf-8

import os, sys, json, cv2, random
import glob
import json
import re
import pickle
import platform
import socket
import hashlib
import logging

import numpy as np
import matplotlib.pyplot as plt
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# ====================

Categories = ['fabric', 'wood', 'plastic', 'metal', 'glass']
# Category_Ids = {'fabric': 0, 'wood': 1, 'plastic': 2, 'metal': 3, 'glass': 4}
Category_Ids = {c: i for i, c in enumerate(Categories)}
Id_Categories = {i: c for i, c in enumerate(Categories)}


def get_indoor_scene_dicts(data_dir='../data/indoor-scene/train'):
    data_dir = os.path.abspath(data_dir)
    json_paths = glob.glob(f'{data_dir}/*.json')

    data_dicts = []
    for idx, json_path in enumerate(json_paths):
        with open(json_path, 'r') as fp:
            j = json.load(fp)
        record = {}
        record['file_name'] = json_path[:-4] + 'jpg'
        record["image_id"] = idx
        record['width'] = j['imageWidth']
        record['height'] = j['imageHeight']

        annotations = []
        for shape in j['shapes']:
            # shape['points']: [[x1,y1],...,[xn,yn]]
            seg1 = [x for xy in shape['points'] for x in xy]  # we only have one seg mask for each instance
            xs, ys = zip(*shape['points'])
            bbox = [min(xs), min(ys), max(xs), max(ys)]
            bbox_mode = BoxMode.XYXY_ABS
            category_id = Category_Ids[shape['label'][:-2]]
            annotations.append(
                {'bbox': bbox, 'bbox_mode': bbox_mode, 'category_id': category_id, 'segmentation': [seg1]})

        record['annotations'] = annotations
        data_dicts.append(record)

    return data_dicts


def register_dataset():
    DatasetCatalog.register('indoor_scene_train', lambda: get_indoor_scene_dicts('../data/indoor-scene/train'))
    MetadataCatalog.get('indoor_scene_train').thing_classes = Categories

    DatasetCatalog.register('indoor_scene_val', lambda: get_indoor_scene_dicts('../data/indoor-scene/val'))
    MetadataCatalog.get('indoor_scene_val').thing_classes = Categories
    # MetadataCatalog.get('indoor_scene_val').evaluator_type = 'coco'

    metadata_train = MetadataCatalog.get("indoor_scene_train")
    metadata_val = MetadataCatalog.get("indoor_scene_val")
    return metadata_train, metadata_val


if __name__ == '__main__':
    # === verify the data loading
    register_dataset()
    metadata_train = MetadataCatalog.get("indoor_scene_train")
    dataset_dicts = get_indoor_scene_dicts()
    for i, d in enumerate(random.sample(dataset_dicts, 5)):
        img = cv2.imread(d["file_name"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        visualizer = Visualizer(img, metadata=metadata_train)
        out = visualizer.draw_dataset_dict(d)
        plt.imshow(out.get_image())
        plt.savefig(f'{i}.jpg')
