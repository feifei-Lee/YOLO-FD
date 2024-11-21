from ultralytics.yolo.utils import DEFAULT_CFG
from ultralytics.yolo.v8.detect import DetectionPredictor


def predict(cfg=DEFAULT_CFG, use_python=False):
    """Runs YOLO model inference on input image(s)."""
    # model = cfg.model or 'yolov8n.pt'
    model = 'yolo-fd.pt'
    source = './demo/v1.mp4'
    args = dict(model=model, source=source)
    if use_python:
        from ultralytics import YOLO
        YOLO(model)(**args)
    else:
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()


if __name__ == '__main__':
    predict()