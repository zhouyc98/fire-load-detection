#!/usr/bin/python3.7

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


Categories = ['fabric', 'wood', 'plastic', 'metal', 'glass']
# Category_Ids = {'fabric': 0, 'wood': 1, 'plastic': 2, 'metal': 3, 'glass': 4}
Category_Ids = {c: i for i, c in enumerate(Categories)}
Id_Categories = {i: c for i, c in enumerate(Categories)}


def get_indoor_scene_dicts(data_dir='../data/indoor-scene/trainval835/', trainval='train'):
    data_dir += trainval
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
    DatasetCatalog.register('indoor_scene_train', lambda: get_indoor_scene_dicts(trainval='train'))
    MetadataCatalog.get('indoor_scene_train').thing_classes = Categories

    DatasetCatalog.register('indoor_scene_val', lambda: get_indoor_scene_dicts(trainval='val'))
    MetadataCatalog.get('indoor_scene_val').thing_classes = Categories
    # MetadataCatalog.get('indoor_scene_val').evaluator_type = 'coco'

    metadata_train = MetadataCatalog.get("indoor_scene_train")
    metadata_val = MetadataCatalog.get("indoor_scene_val")
    return metadata_train, metadata_val


def visualize_all():
    metadata_train, metadata_val = register_dataset()
    data_dir='../data/indoor-scene/trainval1025/'
    for trainval in ('val', 'train'):
        dataset_dicts=get_indoor_scene_dicts(data_dir, trainval)
        for d in dataset_dicts:
            img = cv2.imread(d["file_name"]) # it is full path
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            visualizer = Visualizer(img, metadata=metadata_train)
            out = visualizer.draw_dataset_dict(d)
            plt.imshow(out.get_image())
            vispath=f'{d["file_name"][:-4]}-label.jpg'
            plt.savefig(vispath)
            print('visualize saved:', vispath)


if __name__ == '__main__':
    visualize_all()
