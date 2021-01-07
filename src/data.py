#!/usr/bin/python3.7

import os, sys, json, cv2, random
import glob
import json

import numpy as np
import matplotlib.pyplot as plt
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


# Categories = ['fabric', 'wood', 'plastic', 'metal', 'glass']
Categories = ['fabric', 'wood', 'plastic', 'glass']
Category_Ids = {c: i for i, c in enumerate(Categories)}
Id_Categories = {i: c for i, c in enumerate(Categories)}


def get_indoor_scene_dicts(data_dir='../data/indoor-scene/trainval884/', trainval='train', fold=0):
    """fold: 0-4"""
    json_paths = glob.glob(f'{data_dir}/*.json')
    json_paths.sort()
    
    k_fold = 5
    n = len(json_paths)
    n1 = int(n/k_fold)
    i1 = n1 * fold
    i2 = i1 + n1 if fold < (k_fold-1) else n
    if trainval=='val':
        json_paths=json_paths[i1:i2]
    elif trainval=='train':
        del json_paths[i1:i2]
    else:
        raise AssertionError("trainval should be 'train' or 'val'")

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
            c = shape['label'][:-2]
            if c not in Categories:
                continue
            category_id = Category_Ids[c]
            # shape['points']: [[x1,y1],...,[xn,yn]]
            seg1 = [x for xy in shape['points'] for x in xy]  # we only have one seg mask for each instance
            xs, ys = zip(*shape['points'])
            bbox = [min(xs), min(ys), max(xs), max(ys)]
            bbox_mode = BoxMode.XYXY_ABS
            annotations.append({'bbox': bbox, 'bbox_mode': bbox_mode, 'category_id': category_id, 'segmentation': [seg1]})

        record['annotations'] = annotations
        data_dicts.append(record)

    return data_dicts


def register_dataset(fold=0):
    DatasetCatalog.register('indoor_scene_train', lambda: get_indoor_scene_dicts(trainval='train', fold=fold))
    MetadataCatalog.get('indoor_scene_train').thing_classes = Categories

    DatasetCatalog.register('indoor_scene_val', lambda: get_indoor_scene_dicts(trainval='val', fold=fold))
    MetadataCatalog.get('indoor_scene_val').thing_classes = Categories
    # MetadataCatalog.get('indoor_scene_val').evaluator_type = 'coco'

    metadata_train = MetadataCatalog.get("indoor_scene_train")
    metadata_val = MetadataCatalog.get("indoor_scene_val")
    return metadata_train, metadata_val


def visualize_all(data_dir='../data/indoor-scene/trainval1025/'):
    metadata_train, metadata_val = register_dataset()
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


def _test():
    metadata_train, metadata_val = register_dataset()
    for fold in range(5):
        train_dataset_dicts=get_indoor_scene_dicts(trainval='train',fold=fold)
        val_dataset_dicts=get_indoor_scene_dicts(trainval='val',fold=fold)
        print(f'fold={fold}, n_train={len(train_dataset_dicts)}, n_val={len(val_dataset_dicts)}')
    
    # === result
    # fold=0, n_train=708, n_val=176
    # fold=1, n_train=708, n_val=176
    # fold=2, n_train=708, n_val=176
    # fold=3, n_train=708, n_val=176
    # fold=4, n_train=704, n_val=180


if __name__ == '__main__':
    # _test()
    # visualize_all()
    pass
