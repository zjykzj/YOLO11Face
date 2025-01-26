# -*- coding: utf-8 -*-

"""
@Time    : 2025/1/19 16:32
@File    : eval.py
@Author  : zj
@Description:

Usage - eval using YOLOv5/YOLOv8:
    $ python3 yolo8face_eval.py --model ./runs/yolov5su_widerface.pt --data ./yolo8face/cfg/datasets/widerface.yaml --device 0
    $ python3 yolo8face_eval.py --model ./runs/yolov8s_widerface.pt --data ./yolo8face/cfg/datasets/widerface.yaml --device 0

"""

import yolo8face_utils
from yolo8face_utils import parse_args

from ultralytics.models.yolo.detect import DetectionValidator


def main():
    overrides = parse_args()
    overrides['mode'] = 'val'

    # 初始化验证器并开始评估
    validator = DetectionValidator(args=overrides)
    validator()


if __name__ == "__main__":
    main()
