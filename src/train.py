import os, sys, json, cv2, random
from datetime import datetime

import argparse
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
from detectron2.utils.visualizer import ColorMode
from detectron2.data import build_detection_test_loader
from detectron2.modeling import build_model

from data import *


class Trainer(DefaultTrainer):

    def __init__(self, cfg):
        super().__init__(cfg)
        # self.start_iter = 1
        self.max_iter -= 1  # avoid eval twice in the last

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return COCOEvaluator("indoor_scene_val", ("segm",), False, output_dir=cfg.OUTPUT_DIR + '/eval')

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
        n1 = {'ms7c98-ubuntu': 'S', 'hsh406-ubuntu': 'Z', 'dell-poweredge-t640': 'D'}[socket.gethostname().lower()]
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(f"runs/{n1} {datetime.now().strftime('%m%d-%H%M')} {model_fullname}"),
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
        # logger.info("Model:\n{}".format(model))
        return model


def visualize(model_path='model_final.pth', thr_test=0.5, output_dir='./vis', n=6):
    os.makedirs(output_dir, exist_ok=True)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_path)  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thr_test  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    val_dataset_dicts = get_indoor_scene_dicts('../data/indoor-scene/val')
    if n < 0:
        n = len(val_dataset_dicts)
    for d in random.sample(val_dataset_dicts, n):
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)  # https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        v = Visualizer(img, metadata=metadata_val, scale=1.0)  # instance_mode=ColorMode.IMAGE_BW
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        fn = os.path.split(d["file_name"])[1]
        plt.imshow(out.get_image())
        plt.savefig(f'{output_dir}/{fn[:-4]}-pred.jpg')


def evaluate():
    result = trainer.test(cfg, trainer.model, [COCOEvaluator("indoor_scene_val", ("bbox", "segm",),
                                                             False, output_dir=cfg.OUTPUT_DIR + '/eval')])
    return result, result['segm']['AP']


def get_model_cfg(model_name):
    pre = 'COCO-InstanceSegmentation/mask_rcnn_'
    post = '_3x.yaml'
    model_cfg_dict = {'R50': 'R_50_FPN', 'R101': 'R_101_FPN', 'X101': 'X_101_32x8d_FPN',
                      'R101C4': 'R_101_C4', 'R101DC5': 'R_101_DC5'}

    return pre + model_cfg_dict[model_name] + post


def get_args():
    parser = argparse.ArgumentParser(description='Indoor Fire Load Detection')

    host_name = socket.gethostname().lower()
    _bs = {'ms7c98-ubuntu': 32, 'hsh406-ubuntu': 6, 'dell-poweredge-t640': 10}[host_name]
    _c = '1' if host_name == 'dell-poweredge-t640' else '0'

    parser.add_argument('-n', '--name', type=str, default='R101', help='model name')
    parser.add_argument('-i', '--iter', type=str, default='1k', help='num of training iterations, k=*1000')
    parser.add_argument('-b', '--batch_size', type=int, default=_bs, help='batch size')
    parser.add_argument('-l', '--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('-c', '--cuda', type=str, default=_c, help='cuda visible device id')
    parser.add_argument('-r', '--resume', action='store_true', help='resume training')
    parser.add_argument('-g', '--gamma', type=float, default=0.1, help='lr gamma')
    parser.add_argument('-s', '--step', type=int, default=100000, help='lr decrease step')
    parser.add_argument('--eval_only', action='store_true', help='eval model and exit')
    # parser.add_argument('--fp16', type=int, default=1, help="FP16 acceleration, use 0/1 for false/true")
    # Requires pytorch>=1.6 to use native fp 16 acceleration (https://pytorch.org/docs/stable/notes/amp_examples.html)

    args_ = parser.parse_args()
    args_.iter = int(args_.iter.replace('k', '000'))

    return args_


def rename_model_files(ap_thr_rm=20):
    model_final_path = cfg.OUTPUT_DIR + '/model_final.pth'
    i = str(round(trainer.iter / 1000)) + 'k'
    shutil.copy(model_final_path, cfg.OUTPUT_DIR + f'/{model_fullname}-it{i}-ap{ap:.1f}.pth')

    fns = glob.glob(cfg.OUTPUT_DIR + '/model_0*.pth')
    for fn in fns:
        fn1, fn2 = fn.split('_')
        i = str(round(int(fn2[:-4]) / 1000)) + 'k'
        os.rename(fn, model_final_path)
        _, ap_ = evaluate()
        if ap_ < ap_thr_rm:
            with open(model_final_path, 'r+') as fp:
                fp.truncate()
        os.rename(model_final_path, cfg.OUTPUT_DIR + f'/{model_fullname}-it{i}-ap{ap_:.1f}.pth')


if __name__ == "__main__":
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    metadata_train, metadata_val = register_dataset()
    model_cfg = get_model_cfg(args.name)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_cfg))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_cfg)
    cfg.DATASETS.TRAIN = ('indoor_scene_train',)
    cfg.DATASETS.TEST = ('indoor_scene_val',)
    cfg.OUTPUT_DIR = './output' + args.cuda
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    cfg.SOLVER.AMP.ENABLED = True
    cfg.SOLVER.MAX_ITER = args.iter  # epochs = batch_size * iter / n_images
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size  # global batch_size
    cfg.TEST.EVAL_PERIOD = 500
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.GAMMA = args.gamma
    cfg.SOLVER.STEPS = (args.step,)
    cfg.SOLVER.WARMUP_ITERS = 200
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    _lr = (str(args.lr * 1000) + '/k').replace('.0/k', '/k')
    _r = '-r' if args.resume else ''
    _s = 's' if args.step < args.iter else ''
    model_fullname = f"{args.name}-bs{args.batch_size:02d}-lr{_s}{_lr}{_r}".replace('e-0', 'e-')
    logger = setup_logger(cfg.OUTPUT_DIR + '/log.log')
    logger.info('\n' + '#' * 80 + '\n')
    logger.info('Args: ' + str(args))
    logger.info('Model full name: ' + model_fullname)
    logger.info('Model cfg: ' + model_cfg)

    trainer = Trainer(cfg)
    if args.eval_only:
        trainer.resume_or_load(resume=True)
        evaluate()
        visualize(n=6)
        exit()

    logger.info(f'==================== Start training [{model_fullname}] ====================')
    trainer.resume_or_load(resume=args.resume)
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info('==================== KeyboardInterrupt, early stop ====================')
        pass
    res, ap = evaluate()
    rename_model_files()
    visualize(n=6)
