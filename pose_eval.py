# -*- coding: utf-8 -*-

"""
@Time    : 2025/2/02 16:32
@File    : eval.py
@Author  : zj
@Description:

Usage - eval using YOLOv8-pose:
    $ python3 pose_eval.py --model yolov8n-pose_widerface.pt --data ./yolo8face/cfg/datasets/widerface-landmarks.yaml --device 0
    $ python3 pose_eval.py --model yolov8s-pose_widerface.pt --data ./yolo8face/cfg/datasets/widerface-landmarks.yaml --device 0

"""

import yolo8face_utils
from yolo8face_utils import parse_args

from ultralytics.models.yolo.pose import PoseValidator


def main():
    overrides = parse_args()
    assert overrides['model'] is not None, 'model must be specified'
    assert overrides['data'] is not None, 'data must be specified'
    overrides['mode'] = 'val'

    # 初始化验证器并开始评估
    validator = PoseValidator(args=overrides)
    validator()


if __name__ == "__main__":
    main()
