# -*- coding: utf-8 -*-

"""
@Time    : 2025/2/02 16:32
@File    : eval.py
@Author  : zj
@Description:

Usage - eval using YOLOv8-pose / YOLO11-pose:
    $ python3 pose_eval.py --model yolov8n-pose_widerface.pt
                                   yolov8s-pose_widerface.pt
                                   yolo11n-pose_widerface.pt
                                   yolo11s-pose_widerface.pt --data ./yolo8face/cfg/datasets/widerface-landmarks.yaml --imgsz 640
                                                                                                                              800 --device 0


"""

import yolo11face_utils
from yolo11face_utils import parse_args

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
