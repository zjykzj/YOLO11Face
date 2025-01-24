# -*- coding: utf-8 -*-

"""
@Time    : 2025/1/19 16:51
@File    : widerface_predict.py
@Author  : zj
@Description:

Usage - predict using YOLOv5/YOLOv8:
    $ python3 yolo8face_predict.py --model ./runs/yolov5nu_widerface.pt --source ./yolo8face/assets/ --device 0
    $ python3 yolo8face_predict.py --model ./runs/yolov8_widerface.pt --source ./yolo8face/assets/ --device 0

"""

import yolo8face_utils
from yolo8face_utils import parse_args

from ultralytics.models.yolo.detect import DetectionPredictor



def main():
    overrides = parse_args()
    overrides['mode'] = 'predict'

    # 初始化预测器并开始预测
    predictor = DetectionPredictor(overrides=overrides)
    predictor.predict_cli()


if __name__ == "__main__":
    main()
