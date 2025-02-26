# -*- coding: utf-8 -*-

"""
@Time    : 2025/1/19 16:51
@File    : widerface_predict.py
@Author  : zj
@Description:

Usage - predict using YOLOv5/YOLOv8/YOLO11:
    $ python3 detect_predict.py --model yolov5su_widerface.pt --source ./yolo11face/assets/widerface_val/ --imgsz 640 --device 0
                                        yolov8s_widerface.pt
                                        yolo11s_widerface.pt

"""

import yolo11face_utils
from yolo11face_utils import parse_args

from ultralytics.models.yolo.detect import DetectionPredictor


def main():
    overrides = parse_args()
    assert overrides['model'] is not None, 'model must be specified'
    assert overrides['source'] is not None, 'source must be specified'
    overrides['mode'] = 'predict'

    # 初始化预测器并开始预测
    predictor = DetectionPredictor(overrides=overrides)
    predictor.predict_cli()


if __name__ == "__main__":
    main()
