#!/usr/bin/python3.7

import os, sys, json, random, pickle
from datetime import datetime
from pprint import pprint
from glob import glob

import argparse
import cv2
import socket
import shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
import detectron2
import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import build_detection_test_loader
from detectron2.modeling import build_model, GeneralizedRCNNWithTTA

import data
from data import register_dataset, get_dataset_dicts


class Trainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return COCOEvaluator(f"{args.dataset}_val", ("segm",), False, output_dir=cfg.OUTPUT_DIR + '/eval')

    def build_writers(self):
        """
        Build a list of writers to be used. By default it contains
        writers that write metrics to the screen,
        a json file, and a tensorboard event file respectively.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.

        It is now implemented by:
        ::
            return [
                CommonMetricPrinter(self.max_iter),
                JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
                TensorboardXWriter(self.cfg.OUTPUT_DIR),
            ]

        """
        # Here the default print/log frequency of each writer is used.
        # n1 = {'ms7c98-ubuntu': 'S', 'hsh406-zyc-ubuntu': 'Z', 'dell-poweredge-t640': 'D', 'quincy-ubuntu': 'Y'}[
        #     socket.gethostname().lower()]
        dt_now = datetime.now().strftime('%m%d-%H%M')
        assert not os.path.exists(f"./output/runs/{dt_now} {model_fullname}"), 'runs dir exits!'
        with open(f'{cfg.OUTPUT_DIR}/metrics.json', 'a') as fp:
            fp.write(f'\n# [{dt_now}] {model_fullname} ==========\n')
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(f"./output/runs/{dt_now} {model_fullname}"),
        ]

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        # logger = logging.getLogger(__name__)
        # logger.info("Model:\n{}".format(model)) # suppress this too long message
        return model

    def resume_or_load(self, resume):
        if resume:
            with open(f'{cfg.OUTPUT_DIR}/last_checkpoint', 'r') as fp:
                path = f'{cfg.OUTPUT_DIR}/{fp.read().strip()}'
            logger.info('resume from last_checkpoint: '+path)
        return super().resume_or_load(resume=resume)

def visualize_preds(output_dir='./output/preds'):
    with open(f'{cfg.OUTPUT_DIR}/last_checkpoint', 'r') as fp:
        path = f'{cfg.OUTPUT_DIR}/{fp.read().strip()}'
    logger.info('visualize preds based on last_checkpoint: '+path)
    cfg.MODEL.WEIGHTS = path  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.thr_test  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    
    val_dataset_dicts = get_dataset_dicts(args.dataset, trainval='val', fold=args.fold)
    n_sample=4
    if args.vis_all_preds:
        n_sample = len(val_dataset_dicts)
        output_dir += f'/{model_fullname}-ap{ap:.1f}-thr{args.thr_test}'
    else:
        val_dataset_dicts=random.sample(val_dataset_dicts, n_sample)
    os.makedirs(output_dir, exist_ok=True)
    for i, d in enumerate(val_dataset_dicts):
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)  # https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        v = Visualizer(img, metadata=metadata_val, scale=1.0)  # instance_mode=ColorMode.IMAGE_BW
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        fn = os.path.split(d["file_name"])[1]
        plt.imshow(out.get_image())
        plt.savefig(f'{output_dir}/{fn[:-4]}-pred.jpg')
        logger.info(f'visualize pred saved ({i+1}/{n_sample}): {output_dir}/{fn[:-4]}-pred.jpg')


def rename_models():
    model_final_path = cfg.OUTPUT_DIR + '/model_final.pth'
    with open(f'{cfg.OUTPUT_DIR}/last_checkpoint', 'w') as fp:
        fp.write('model_final.pth')
    with open(cfg.OUTPUT_DIR+'/metrics.json','r') as fp:
        metrics = fp.readlines()
    ind = [i for i,m in enumerate(metrics) if m[0]=='#'][-1]
    metrics = [m for m in metrics[ind+1:] if 'segm/AP' in m]
    d = [json.loads(m) for m in metrics]
    d = dict([(d1['iteration'], d1['segm/AP']) for d1 in d]) # {iter: ap, }
    print('metrics:')
    pprint(d)

    ap_i_fns = []
    for fn in sorted(glob(cfg.OUTPUT_DIR + '/model_0*.pth')):
        i = int(fn.split('model_0')[1][:-4])
        ap = d[i]
        ik = f'{round(i / 1000, 1)}k'
        fn_new = f'{cfg.OUTPUT_DIR}/{model_fullname}-it{ik}-ap{ap:.1f}.pth'
        ap_i_fns.append((ap, i, fn_new))
        os.rename(fn, fn_new)
        logger.info(f'rename: {fn} -> {fn_new}')

    # save best model
    ap_i_fns.sort()
    ap_max, i_max, fn_max=ap_i_fns[-1]
    logger.info(f'save as model_final: {fn_max}')
    shutil.copy(fn_max, model_final_path)
    del ap_i_fns[-1]
    
    # save step model
    for idx in range(len(ap_i_fns)-1,-1,-1):
        ap, i, fn = ap_i_fns[idx]
        if i+1 == cfg.SOLVER.STEPS[0]:
            fn_new = fn.replace('-lrs','-lr')
            logger.info(f"save step model: {fn_new} (lrs -> lr)")
            os.rename(fn, fn_new)
            del ap_i_fns[idx]
    
    # clear models
    if not args.save_all:
        logger.info('clear models ...')
        for _, i, fn in ap_i_fns:
            os.remove(fn)
    
    # eval with tta for model_final
    if args.tta:
        logger.info(f'===== Eval with TTA: {fn_max}')
        _, ap = evaluate(resume=True, tta=True)
        logger.info('Eval AP (TTA): ' + str(ap))
        os.rename(fn_max, f'{fn_max[:-4]}-aap{ap:.2f}.pth')

    
    return ap_max, fn_max


def evaluate(resume=False, tta=False):
    if resume:
        trainer.resume_or_load(resume=True)
    if tta:
        cfg.TEST.AUG.ENABLED = True
        trainer.model=GeneralizedRCNNWithTTA(cfg, trainer.model)
    
    result = trainer.test(cfg, trainer.model, 
                [COCOEvaluator(f"{args.dataset}_val", ("bbox", "segm",), distributed=False, output_dir=cfg.OUTPUT_DIR + '/eval')])
    
    return result, result['segm']['AP']


def get_model_cfg(cfg, model_name):
    if model_name[-1] !='m':
        if model_name=='X152':
            model_cfg_path =  'Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml'
        else:
            pre = 'COCO-InstanceSegmentation/mask_rcnn_'
            post = '_1x.yaml' if model_name.endswith('_1x') else '_3x.yaml'
            model_cfg_dict = {'R50': 'R_50_FPN', 'R50C4': 'R_50_C4', 'R50DC5': 'R_50_DC5', 'R50_1x': 'R_50_FPN',
                            'R101': 'R_101_FPN', 'R101C4': 'R_101_C4', 'R101DC5': 'R_101_DC5',
                            'X101': 'X_101_32x8d_FPN'}
            model_cfg_path = pre + model_cfg_dict[model_name] + post
        cfg.merge_from_file(model_zoo.get_config_file(model_cfg_path))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_cfg_path)
    else:
        # pretrained in MIT67 dataset
        cfg = get_model_cfg(cfg, model_name[:-1])
        cfg.MODEL.WEIGHTS = sorted(glob(f'../models/{model_name[:-1]}-m*.pkl'))[-1]
    
    return cfg


def get_args():
    parser = argparse.ArgumentParser(description='Indoor Fire Load Detection')

    parser.add_argument('-m', '--model_name', type=str, default='R50', help='model name')
    parser.add_argument('-d', '--dataset', type=str, default='indoor_scene', help='dataset name')
    parser.add_argument('-f', '--fold', type=int, default=2, help='dataset fold')
    parser.add_argument('-i', '--iter', type=str, default='1k', help='num of training iterations, k=*1000')
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('-l', '--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('-c', '--cuda', type=str, default='', help='cuda visible device id')
    parser.add_argument('-r', '--resume', action='store_true', help='resume training')
    parser.add_argument('-g', '--gamma', type=float, default=0.1, help='lr gamma')
    parser.add_argument('-s', '--step', type=str, default='100k', help='lr decrease step')
    parser.add_argument('--step2', type=str, default='200k', help='lr decrease step2')
    parser.add_argument('--step3', type=str, default='300k', help='lr decrease step3')
    parser.add_argument('--eval_only', action='store_true', help='eval model and exit')
    parser.add_argument('--save_all', action='store_true', help='save all checkpoints in model training')
    parser.add_argument('--to_pkl', action='store_true', help='convert model to pkl format and exit')
    parser.add_argument('--vis_all_preds', action='store_true', help='visualize all preds for val dataset')
    parser.add_argument('--tta', action='store_true', help='test time augmentation')
    parser.add_argument('--thr_test', type=float, default=0.5, help='ROI thr test')
    parser.add_argument('--fp16', type=int, default=2, help="FP16 acceleration, use 0/1/2 for false/true/auto")
    # Requires pytorch>=1.6 to use native fp 16 acceleration (https://pytorch.org/docs/stable/notes/amp_examples.html)

    args_ = parser.parse_args()
    
    if args_.fp16==2:
        args_.fp16 = 0 if args_.model_name=='X152' else 1
    args_.fp16 = bool(args_.fp16)
    assert args_.iter[-1] == 'k' and args_.step[-1] == 'k' and args_.step2[-1] == 'k' and args_.step3[-1] == 'k'
    args_.iter = int(float(args_.iter[:-1]) * 1000)
    args_.step = int(float(args_.step[:-1]) * 1000)
    args_.step2 = int(float(args_.step2[:-1]) * 1000)
    args_.step3 = int(float(args_.step3[:-1]) * 1000)
    host_name = socket.gethostname().lower()
    if not args_.cuda:
        args_.cuda = '1' if host_name == 'dell-poweredge-t640' else '0'

    return args_


if __name__ == "__main__":
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    metadata_train, metadata_val = register_dataset(args.dataset, fold=args.fold)
    
    cfg = get_cfg()
    cfg = get_model_cfg(cfg, args.model_name)
    cfg.DATASETS.TRAIN = (f'{args.dataset}_train',)
    cfg.DATASETS.TEST = (f'{args.dataset}_val',)
    cfg.OUTPUT_DIR = './output' if args.cuda=='0' else './output'+args.cuda
    cfg.SEED = 7
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(data.Categories) # const var in data.py
    cfg.SOLVER.AMP.ENABLED = args.fp16
    cfg.SOLVER.MAX_ITER = args.iter  # epochs = batch_size * iter / n_images
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size  # global batch_size
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.GAMMA = args.gamma
    cfg.SOLVER.STEPS = (args.step, args.step2, args.step3)
    cfg.SOLVER.WARMUP_ITERS = 100
    cfg.TEST.EVAL_PERIOD = 200 if args.batch_size < 4 else 100
    cfg.SOLVER.CHECKPOINT_PERIOD = cfg.TEST.EVAL_PERIOD
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.33, 1.0, 3.0]] # slightly better than default
    cfg.INPUT.CROP.ENABLED = True

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    shutil.rmtree(cfg.OUTPUT_DIR+'/eval', ignore_errors=True) # remove cache
    _lr = f'{args.lr * 1000}x'  # lr 1x = 1/1000
    _r = '-r' if args.resume else ''
    _s = 's' if args.step < args.iter else ''
    _d = '-'+args.dataset[0] if args.dataset != 'indoor_scene' else ''
    model_fullname = f"{args.model_name}{_d}-f{args.fold}-bs{args.batch_size:02d}-lr{_s}{_lr}{_r}".replace('e-0', 'e-')
    logger = setup_logger(cfg.OUTPUT_DIR + '/log.log')
    logger.info('#' * 100 + '\n')
    logger.info('Args: ' + str(args))
    logger.info('Model weights: ' + cfg.MODEL.WEIGHTS)
    logger.info('Model full name: ' + model_fullname)

    trainer = Trainer(cfg)
    if args.eval_only:
        # eval model_final
        result, ap = evaluate(resume=True, tta=args.tta)
        visualize_preds()
        exit()
    if args.to_pkl:
        # resume model_final and save it to pkl
        trainer.resume_or_load(resume=True)
        with open('model_final.pkl','wb') as fp:
            pickle.dump({'model':trainer.model.state_dict(), '__author__':'zhou-yucheng'}, fp)
        exit()
    trainer.resume_or_load(resume=args.resume)
    if args.resume:
        # lr and optimizer will also be resumed, change lr by using steps
        trainer.max_iter = args.iter # + trainer.start_iter
        trainer.scheduler.milestones = cfg.SOLVER.STEPS
    
    logger.info(f'==================== Start training [{model_fullname}] ====================')
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info('==================== KeyboardInterrupt, early stop ====================')
        args.tta = False
        pass
    ap, fn = rename_models() # ap will be used in visualize_preds
    visualize_preds()
