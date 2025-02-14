# -*- coding: utf-8 -*-

"""
@Time    : 2025/2/02 16:31
@File    : train.py
@Author  : zj
@Description:

Usage - training using YOLOv8-pose:
    $ python3 pose_train.py --model yolov8n-pose.pt --data ./yolo8face/cfg/datasets/widerface-landmarks.yaml --epochs 100 --imgsz 640 --device 0
    $ python3 pose_train.py --model yolov8s-pose.pt --data ./yolo8face/cfg/datasets/widerface-landmarks.yaml --epochs 100 --imgsz 640 --device 0

"""

import yolo8face_utils
from yolo8face_utils import parse_args

from ultralytics.models.yolo.pose import PoseTrainer


def main():
    overrides = parse_args()
    assert overrides['model'] is not None, 'model must be specified'
    assert overrides['data'] is not None, 'data must be specified'
    overrides['mode'] = 'train'

    # 初始化训练器并开始训练
    trainer = PoseTrainer(overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()
