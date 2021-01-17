#!/usr/bin/python3.7

import os, sys, json, cv2, random
import glob
import json
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


# Categories = ['fabric', 'wood', 'plastic', 'metal', 'glass']
Categories = ['fabric', 'wood', 'plastic', 'glass']
Category_Ids = {c: i for i, c in enumerate(Categories)}
Id_Categories = {i: c for i, c in enumerate(Categories)}


def get_indoor_scene_dicts(data_dir='../data/indoor-scene/trainval1k/', trainval='train', fold=2):
    """ fold: 0-4 (fold=2 is better)
        return: list[dict]
        ref: https://detectron2.readthedocs.io/tutorials/datasets.html
    """
    json_paths = glob.glob(f'{data_dir}/*.json')
    json_paths.sort()

    # json_paths=np.array(json_paths)
    # np.random.seed(0)
    # np.random.shuffle(json_paths)
    # json_paths=list(json_paths)
    
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


def register_dataset(fold=2):
    DatasetCatalog.register('indoor_scene_train', lambda: get_indoor_scene_dicts(trainval='train', fold=fold))
    MetadataCatalog.get('indoor_scene_train').thing_classes = Categories

    DatasetCatalog.register('indoor_scene_val', lambda: get_indoor_scene_dicts(trainval='val', fold=fold))
    MetadataCatalog.get('indoor_scene_val').thing_classes = Categories
    # MetadataCatalog.get('indoor_scene_val').evaluator_type = 'coco'

    metadata_train = MetadataCatalog.get("indoor_scene_train")
    metadata_val = MetadataCatalog.get("indoor_scene_val")
    return metadata_train, metadata_val


def visualize_all_label(data_dir='../data/indoor-scene/trainval1k/'):
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


def fold_stat():
    print(Categories)
    metadata_train, metadata_val = register_dataset()
    for fold in range(5):
        train_dataset_dicts=get_indoor_scene_dicts(trainval='train',fold=fold)
        val_dataset_dicts=get_indoor_scene_dicts(trainval='val',fold=fold)
        print(f'fold={fold}, n_train={len(train_dataset_dicts)}, n_val={len(val_dataset_dicts)}')

        N=[[0,0,0,0], [0,0,0,0]]
        for i, dataset_dicts in enumerate((train_dataset_dicts, val_dataset_dicts)):
            for record in dataset_dicts:
                for ann in record['annotations']:
                    N[i][ann['category_id']] += 1
        print('train:', N[0])
        print('val:', N[1])
        n = np.array([round(i/j,2) for i,j in zip(*N)])
        print(f'ratio: {n} ({n.mean():.4f} ± {n.std():.3f})', end='\n\n')

    # === result (trainval1k)
        # ['fabric', 'wood', 'plastic', 'glass']
        # fold=0, n_train=812, n_val=203
        # train: [2576, 2269, 854, 1447]
        # val: [600, 554, 223, 335]
        # ratio: [4.29 4.1  3.83 4.32] (4.1350 ± 0.195)

        # fold=1, n_train=812, n_val=203
        # train: [2504, 2237, 879, 1417]
        # val: [672, 586, 198, 365]
        # ratio: [3.73 3.82 4.44 3.88] (3.9675 ± 0.278)

        # fold=2, n_train=812, n_val=203
        # train: [2536, 2281, 855, 1431]
        # val: [640, 542, 222, 351]
        # ratio: [3.96 4.21 3.85 4.08] (4.0250 ± 0.134)

        # fold=3, n_train=812, n_val=203
        # train: [2525, 2278, 853, 1429]
        # val: [651, 545, 224, 353]
        # ratio: [3.88 4.18 3.81 4.05] (3.9800 ± 0.145)

        # fold=4, n_train=812, n_val=203
        # train: [2563, 2227, 867, 1404]
        # val: [613, 596, 210, 378]
        # ratio: [4.18 3.74 4.13 3.71] (3.9400 ± 0.216)


def fold_stat_bbox_area():
    print('large, medium, small')
    metadata_train, metadata_val = register_dataset()
    for fold in range(5):
        train_dataset_dicts=get_indoor_scene_dicts(trainval='train',fold=fold)
        val_dataset_dicts=get_indoor_scene_dicts(trainval='val',fold=fold)
        print(f'fold={fold}, n_train={len(train_dataset_dicts)}, n_val={len(val_dataset_dicts)}')

        N=[[0,0,0], [0,0,0]] # large, medium, small
        for i, dataset_dicts in enumerate((train_dataset_dicts, val_dataset_dicts)):
            for record in dataset_dicts:
                for ann in record['annotations']:
                    x1, y1, x2, y2 = ann['bbox']
                    area = abs((x1-x2)*(y1-y2))
                    if area > 96**2:
                        N[i][0]+=1
                    elif area > 32**2:
                        N[i][1]+=1
                    else:
                        N[i][2]+=1

        print('train:', N[0])
        print('val:', N[1])
        n = np.array([round(i/j,2) for i,j in zip(*N)])
        print(f'ratio: {n} ({n.mean():.4f} ± {n.std():.3f})', end='\n\n')

    # === result
        # large, medium, small
        # fold=0, n_train=820, n_val=204
        # train: [3436, 3402, 414]
        # val: [868, 772, 75]
        # ratio: [3.96 4.41 5.52] (4.6300 ± 0.656)

        # fold=1, n_train=820, n_val=204
        # train: [3409, 3337, 396]
        # val: [895, 837, 93]
        # ratio: [3.81 3.99 4.26] (4.0200 ± 0.185)

        # fold=2, n_train=820, n_val=204
        # train: [3478, 3332, 386]
        # val: [826, 842, 103]
        # ratio: [4.21 3.96 3.75] (3.9733 ± 0.188)

        # fold=3, n_train=820, n_val=204
        # train: [3472, 3327, 386]
        # val: [832, 847, 103]
        # ratio: [4.17 3.93 3.75] (3.9500 ± 0.172)

        # fold=4, n_train=816, n_val=208
        # train: [3421, 3298, 374]
        # val: [883, 876, 115]
        # ratio: [3.87 3.76 3.25] (3.6267 ± 0.270)


def fold_stat_img_class():
    classes=('Bedroom', 'Clothing store', 'Dining room', 'Hospital', 'Living room', 'Office')
    print(classes)
    names=[]
    for c in classes:
        with open(f'../data/indoor-scene/trainval1k-class/{c}.txt','r',encoding='utf-8-sig') as fp:
            names.append(fp.readlines())
    
    metadata_train, metadata_val = register_dataset()
    for fold in range(5):
        train_dataset_dicts=get_indoor_scene_dicts(trainval='train',fold=fold)
        val_dataset_dicts=get_indoor_scene_dicts(trainval='val',fold=fold)
        print(f'fold={fold}, n_train={len(train_dataset_dicts)}, n_val={len(val_dataset_dicts)}')

        N=[[0,0,0,0,0,0], [0,0,0,0,0,0]]
        for i, dataset_dicts in enumerate((train_dataset_dicts, val_dataset_dicts)):
            for record in dataset_dicts:
                fn=os.path.split(record['file_name'])[1]+'\n' # consistent with names
                ci=-1
                for ni,cnames in enumerate(names):
                    if fn in cnames:
                        ci=ni
                        break
                assert ci>=0
                N[i][ci]+=1

        print('train:\t', N[0])
        print('val:\t', N[1])
        n = np.array([round(i/j,2) if j>0 else -1 for i,j in zip(*N)])
        print(f'ratio:\t {list(n)} ({n.mean():.4f} ± {n.std():.3f})', end='\n\n')
    
    # === result
        # ('Bedroom', 'Clothing store', 'Dining room', 'Hospital', 'Living room', 'Office')
        # fold=0, n_train=820, n_val=204
        # train:   [264, 6, 182, 64, 108, 196]
        # val:     [83, 1, 29, 12, 34, 45]
        # ratio:   [3.18, 6.0, 6.28, 5.33, 3.18, 4.36] (4.7217 ± 1.246)

        # fold=1, n_train=820, n_val=204
        # train:   [285, 4, 167, 57, 111, 196]
        # val:     [62, 3, 44, 19, 31, 45]
        # ratio:   [4.6, 1.33, 3.8, 3.0, 3.58, 4.36] (3.4450 ± 1.079)

        # fold=2, n_train=820, n_val=204
        # train:   [279, 6, 173, 61, 113, 188]
        # val:     [68, 1, 38, 15, 29, 53]
        # ratio:   [4.1, 6.0, 4.55, 4.07, 3.9, 3.55] (4.3617 ± 0.790)

        # fold=3, n_train=820, n_val=204
        # train:   [284, 6, 161, 63, 112, 194]
        # val:     [63, 1, 50, 13, 30, 47]
        # ratio:   [4.51, 6.0, 3.22, 4.85, 3.73, 4.13] (4.4067 ± 0.884)

        # fold=4, n_train=816, n_val=208
        # train:   [276, 6, 161, 59, 124, 190]
        # val:     [71, 1, 50, 17, 18, 51]
        # ratio:   [3.89, 6.0, 3.22, 3.47, 6.89, 3.73] (4.5333 ± 1.392)


def _test():
    metadata_train, metadata_val = register_dataset()
    train_dataset_dicts=get_indoor_scene_dicts(trainval='train')
    val_dataset_dicts=get_indoor_scene_dicts(trainval='val')
    ns=[]
    for record in train_dataset_dicts+val_dataset_dicts:
        ns.append(len(record['annotations']))

    ns=np.array(ns)
    hist, bin_edges = np.histogram(ns, bins=range(1,max(ns)+1))
    print('hist:',hist)
    print('bin_edges:',bin_edges)
    print(f'max instance num/img={max(ns)}, in {len(ns)} imgs') # max n=26


if __name__ == '__main__':
    # fold_stat()
    # fold_stat_bbox_area()
    # fold_stat_img_class()
    visualize_all_label()
