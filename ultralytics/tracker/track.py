# Ultralytics YOLO 🚀, AGPL-3.0 license

from functools import partial

import torch

from ultralytics.yolo.utils import IterableSimpleNamespace, yaml_load
from ultralytics.yolo.utils.checks import check_yaml

from .trackers import BOTSORT, BYTETracker
from social_network.ConcatMatrix import ContactMatrix
concate_matrix = ContactMatrix()

TRACKER_MAP = {'bytetrack': BYTETracker, 'botsort': BOTSORT}



def on_predict_start(predictor, persist=False):
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.

    Raises:
        AssertionError: If the tracker_type is not 'bytetrack' or 'botsort'.
    """
    if hasattr(predictor, 'trackers') and persist:
        return
    tracker = check_yaml(predictor.args.tracker)
    cfg = IterableSimpleNamespace(**yaml_load(tracker))
    assert cfg.tracker_type in ['bytetrack', 'botsort'], \
        f"Only support 'bytetrack' and 'botsort' for now, but got '{cfg.tracker_type}'"
    trackers = []
    for _ in range(predictor.dataset.bs):
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)
        trackers.append(tracker)
    predictor.trackers = trackers


def on_predict_postprocess_end(predictor):
    """Postprocess detected boxes and update with object tracking."""
    bs = predictor.dataset.bs
    im0s = predictor.batch[1]
    for i in range(bs):
        det = predictor.results[i].boxes.cpu().numpy()
        if len(det) == 0:
            continue
        tracks, tracklets_tracked = predictor.trackers[i].update(det, im0s[i])
        if len(tracks) == 0:
            continue
        idx = tracks[:, -1].astype(int)
        predictor.results[i] = predictor.results[i][idx]

        # TODO 在此处更改cls的值
        def check_continue_frame(list):
            if len(list) <= 10:  # 指定5帧
                return False
            first_five_elements = list[-11:-1]
            return all(element == first_five_elements[0] for element in first_five_elements)
        if tracklets_tracked is not None:
            history_cls = [t.history_cls for t in tracklets_tracked]
            for his_c in history_cls:
                if check_continue_frame(his_c):
                    his_c[-1] = his_c[-6]  # 保持与前一帧一致
            # 保持最长存储100的历史class
            history_cls = [his_c[-100:] if len(his_c)>100 else his_c for his_c in history_cls ]
            for j, t in enumerate(tracklets_tracked):  # 修改多目标跟踪中的history_cls
                t.history_cls = history_cls[j]

            for j, t in enumerate(tracks):
                t[-2] = history_cls[j][-1]

            # # 添加接触矩阵
            # detection_boxes = tracks[:,:5]
            # concate_matrix.compute_concate(detection_boxes)
            # concate_matrix.print_matrix()
            # concate_matrix.build_graph()

        predictor.results[i].update(boxes=torch.as_tensor(tracks[:, :-1]))
        predictor.results[i].tracklets_tracked = tracklets_tracked  # 将历史的追踪结果添加到results中


def register_tracker(model, persist):
    """
    Register tracking callbacks to the model for object tracking during prediction.

    Args:
        model (object): The model object to register tracking callbacks for.
        persist (bool): Whether to persist the trackers if they already exist.

    """
    model.add_callback('on_predict_start', partial(on_predict_start, persist=persist))
    model.add_callback('on_predict_postprocess_end', on_predict_postprocess_end)
