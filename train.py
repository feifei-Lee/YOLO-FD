from ultralytics.yolo.v8.detect import DetectionTrainer
from ultralytics.yolo.utils import DEFAULT_CFG


def train(cfg=DEFAULT_CFG, use_python=False):
    """Train and optimize YOLO model given training data and device."""
    model = cfg.model
    model = 'yolov8s.yaml'
    data =  'norcardis_disease.yaml'
    # data = 'voc07_12.yaml'
    device = cfg.device if cfg.device is not None else ''
    freeze = 0
    epochs = 300
    pre_weights = None # './yolov8s.pt'
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