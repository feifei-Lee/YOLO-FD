# Ultralytics YOLO 🚀, AGPL-3.0 license
import os
from copy import copy

import numpy as np
import torch

from ultralytics.nn.tasks import DetectionModel
from ultralytics.yolo import v8
from ultralytics.yolo.data import build_dataloader, build_yolo_dataset
from ultralytics.yolo.data.dataloaders.v5loader import create_dataloader, create_seg_dataloader
from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, RANK, colorstr
from ultralytics.yolo.utils.plotting import plot_images, plot_labels, plot_results
from ultralytics.yolo.utils.torch_utils import de_parallel, torch_distributed_zero_first

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# torch.use_deterministic_algorithms(True, warn_only=True)

# BaseTrainer python usage
class DetectionTrainer(BaseTrainer):

    def build_dataset(self, img_path, mode='train', batch=None):
        """Build YOLO Dataset

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == 'val', stride=gs)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        """TODO: manage splits differently."""
        # Calculate stride - check if model is initialized
        if self.args.v5loader:
            # LOGGER.warning("WARNING ⚠️  using v5loader")
            gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
            return create_dataloader(path=dataset_path,
                                     imgsz=self.args.imgsz,
                                     batch_size=batch_size,
                                     stride=gs,
                                     hyp=vars(self.args),
                                     augment=mode == 'train',
                                     cache=self.args.cache,
                                     pad=0 if mode == 'train' else 0.5,
                                     rect=self.args.rect or mode == 'val',
                                     rank=rank,
                                     workers=self.args.workers,
                                     close_mosaic=self.args.close_mosaic != 0,
                                     prefix=colorstr(f'{mode}: '),
                                     shuffle=mode == 'train',
                                     seed=self.args.seed)[0]
        assert mode in ['train', 'val']
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == 'train'
        if getattr(dataset, 'rect', False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == 'train' else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader

    def get_seg_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        if self.args.v5loader:
            # LOGGER.warning("WARNING ⚠️  using v5loader")
            gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
            return create_seg_dataloader(dataset_path,
                                         self.data['segnc'],  # segnc
                                         self.args.imgsz,
                                         batch_size,
                                         gs,
                                         False,  # single_cls,
                                         hyp=vars(self.args),
                                         augment=mode == 'train',
                                         cache=self.args.cache,
                                         rect=self.args.rect,
                                         rank=rank,
                                         workers=self.args.workers,
                                         image_weights=False,
                                         quad=False,  # quad=opt.quad,
                                         pad=0.0 if mode == 'train' else 0.5,
                                         prefix=colorstr(f'{mode}: '),
                                         shuffle=mode == 'train',
                                         seed=self.args.seed)[0]

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255
        return batch

    def set_model_attributes(self):
        """nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)."""
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data['nc']  # attach number of classes to model
        self.model.names = self.data['names']  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = DetectionModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
        return v8.detect.DetectionValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def label_loss_items(self, loss_items=None, prefix='train'):
        """
        Returns a loss dict with labelled training loss items tensor
        """
        # Not needed for classification but necessary for segmentation & detection
        keys = [f'{prefix}/{x}' for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ('\n' + '%11s' *
                (4 + len(self.loss_names))) % ('Epoch', 'GPU_mem', *self.loss_names, 'Instances', 'Size')

    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        plot_images(images=batch['img'],
                    batch_idx=batch['batch_idx'],
                    cls=batch['cls'].squeeze(-1),
                    bboxes=batch['bboxes'],
                    paths=batch['im_file'],
                    fname=self.save_dir / f'train_batch{ni}.jpg',
                    on_plot=self.on_plot)

    def plot_metrics(self):
        """Plots metrics from a CSV file."""
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb['bboxes'] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb['cls'] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data['names'], save_dir=self.save_dir, on_plot=self.on_plot)


def train(cfg=DEFAULT_CFG, use_python=False):
    """Train and optimize YOLO model given training data and device."""
    model = cfg.model
    model = 'yolov8s-swinT.yaml'
    model = 'yolov8s-ContextAggregation.yaml'
    model = 'yolov8s-biformer-1.yaml'
    model = 'yolov8s.yaml'
    # model = 'yolov8n.yaml'
    # model = 'yolov8s-swinT.yaml'
    # model = 'C:\\Users\zhaor\PycharmProjects\\ultralytics-main\\ultralytics\yolo\\v8\detect\\runs\detect\\train605\weights\last.pt'
    data = 'norcardis_disease.yaml'  # voc2012.yaml
    # data = 'voc07_12.yaml'
    device = cfg.device if cfg.device is not None else ''
    freeze = 0
    epochs = 300
    pre_weights = None
    # pre_weights = 'C:\\Users\zhaor\PycharmProjects\\ultralytics-main\pre_weights\\best.pt'  # model的***.pt比pre_weights优先级高，同时存在的情况下，优先使用model的***.pt
    # pre_weights = 'C:\\Users\zhaor\PycharmProjects\\ultralytics-main\yolov8s.pt'
    # pre_weights = 'C:\\Users\zhaor\PycharmProjects\\ultralytics-main\pre_weights\mtl\\uncertainty\weights\\best.pt'
    batch = 8
    mtl = 0  # 0 equal, 1 uncertainty, 2 gradnorm, 3 DWA, -1 by turn
    pcgrad = False
    cagrad = False
    args = dict(model=model,
                data=data,
                device=device,
                pre_weights=pre_weights,
                freeze=freeze,
                epochs=epochs,
                batch=batch,
                mtl=mtl,
                pcgrad=pcgrad,
                cagrad=cagrad)
    if use_python:
        from ultralytics import YOLO
        YOLO(model).train(**args)
    else:
        trainer = DetectionTrainer(overrides=args)
        trainer.train()


if __name__ == '__main__':
    train()
