# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops


class DetectionPredictor(BasePredictor):

    def postprocess(self, preds, img, orig_imgs):
        """Postprocesses predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            path = self.batch[0]
            img_path = path[i] if isinstance(path, list) else path
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results


def predict(cfg=DEFAULT_CFG, use_python=False):
    """Runs YOLO model inference on input image(s)."""
    # model = cfg.model or 'yolov8n.pt'
    # model = 'C:\\Users\\zhaor\\PycharmProjects\\ultralytics-main\\best_207.pt'
    # model = 'C:\\Users\\zhaor\\PycharmProjects\\ultralytics-main\\ultralytics\\yolo\\v8\detect\\runs\\detect\\train205_voc207epoch_mIOU\\weights\\last.pt'
    model = 'C:\\Users\zhaor\PycharmProjects\\ultralytics-main\pre_weights\mtl\pcgrad+uncert\weights\\best.pt'
    # source = cfg.source if cfg.source is not None else ROOT / 'assets' if (ROOT / 'assets').exists() \
    #     else 'https://ultralytics.com/images/bus.jpg'
    source = 'C:\\Users\zhaor\PycharmProjects\\ultralytics-main\\video\\test_video_2.mp4'
    # source = 'C:\\Users\zhaor\Desktop\dataset-fish\\1080p\\test_fps'
    source = 'C:\dataset\\norcardia_disease_fish\det\image-all'
    # source = 'C:\\Users\zhaor\PycharmProjects\\ultralytics-main\datasets\\norcardia_disease_fish\det\images\\train'
    # source = 'C:\\Users\zhaor\Desktop\\result-csv\detection result\\figure_compare\\faster-rcnn'
    # source = 'C:\\Users\zhaor\PycharmProjects\\ultralytics-main\compare_pic\\0909_023002.jpg'
    source = 'C:\\Users\zhaor\PycharmProjects\\ultralytics-main\compare_pic\\0907_014198.jpg'
    args = dict(model=model, source=source)
    if use_python:
        from ultralytics import YOLO
        YOLO(model)(**args)
    else:
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()


if __name__ == '__main__':
    predict()
