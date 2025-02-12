# -*- coding: utf-8 -*-

"""
@Time    : 2025/1/19 16:31
@File    : train.py
@Author  : zj
@Description:

Usage - training using YOLOv5:
    $ python3 detect_train.py --model yolov5nu.pt --data ./yolo8face/cfg/datasets/widerface.yaml --epochs 300 --imgsz 800 --batch 8 --device 0
    $ python3 detect_train.py --model yolov5su.pt --data ./yolo8face/cfg/datasets/widerface.yaml --epochs 300 --imgsz 800 --batch 8 --device 0

Usage - training using YOLOv8:
    $ python3 detect_train.py --model yolov8n.pt --data ./yolo8face/cfg/datasets/widerface.yaml --epochs 300 --imgsz 800 --batch 8 --device 0
    $ python3 detect_train.py --model yolov8s.pt --data ./yolo8face/cfg/datasets/widerface.yaml --epochs 300 --imgsz 800 --batch 8 --device 0

"""

import yolo8face_utils
from yolo8face_utils import parse_args

from ultralytics.models.yolo.detect import DetectionTrainer


def main():
    overrides = parse_args()
    assert overrides['model'] is not None, 'model must be specified'
    assert overrides['data'] is not None, 'data must be specified'
    overrides['mode'] = 'train'

    # 初始化训练器并开始训练
    trainer = DetectionTrainer(overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()
