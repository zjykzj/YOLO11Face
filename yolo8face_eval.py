# -*- coding: utf-8 -*-

"""
@Time    : 2025/1/19 16:32
@File    : eval.py
@Author  : zj
@Description:

Usage - eval using YOLOv5:
    $ python3 yolo8face_eval.py --model yolov5nu_widerface.pt --data ./yolo8face/cfg/datasets/widerface.yaml --imgsz 640 --device 0
    $ python3 yolo8face_eval.py --model yolov5su_widerface.pt --data ./yolo8face/cfg/datasets/widerface.yaml --imgsz 640 --device 0
    $ python3 yolo8face_eval.py --model yolov5su_widerface.pt --data ./yolo8face/cfg/datasets/widerface.yaml --imgsz 800 --device 0

Usage - training using YOLOv8:
    $ python3 yolo8face_train.py --model yolov8n_widerface.pt --data ./yolo8face/cfg/datasets/widerface.yaml --imgsz 640 --device 0
    $ python3 yolo8face_train.py --model yolov8s_widerface.pt --data ./yolo8face/cfg/datasets/widerface.yaml --imgsz 640 --device 0
    $ python3 yolo8face_train.py --model yolov8s_widerface.pt --data ./yolo8face/cfg/datasets/widerface.yaml --imgsz 800 --device 0

"""

import yolo8face_utils
from yolo8face_utils import parse_args

from ultralytics.models.yolo.detect import DetectionValidator


def main():
    overrides = parse_args()
    assert overrides['model'] is not None, 'model must be specified'
    assert overrides['data'] is not None, 'data must be specified'
    overrides['mode'] = 'val'

    # 初始化验证器并开始评估
    validator = DetectionValidator(args=overrides)
    validator()


if __name__ == "__main__":
    main()
