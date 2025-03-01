# -*- coding: utf-8 -*-

"""
@Time    : 2025/1/19 16:32
@File    : eval.py
@Author  : zj
@Description:

Usage - eval using YOLOv5/YOLOv8/YOLO11:
    $ python3 detect_eval.py --model yolov5nu_widerface.pt --data ./yolo11face/cfg/datasets/widerface.yaml --imgsz 640 --device 0
                                     yolov5su_widerface.pt
                                     yolov8s_widerface.pt
                                     yolo11s_widerface.pt
                                                                                                                  
"""

import yolo11face_utils
from yolo11face_utils import parse_args

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
