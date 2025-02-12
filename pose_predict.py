# -*- coding: utf-8 -*-

"""
@Time    : 2025/2/02 16:51
@File    : widerface_predict.py
@Author  : zj
@Description:

Usage - predict using YOLOv8:
    $ python3 pose_predict.py --model yolov8n-pose_widerface.pt --source ./yolo8face/assets/widerface_val/ --device 0

"""

import yolo8face_utils
from yolo8face_utils import parse_args

from ultralytics.models.yolo.pose import PosePredictor


def main():
    overrides = parse_args()
    assert overrides['model'] is not None, 'model must be specified'
    assert overrides['source'] is not None, 'source must be specified'
    overrides['mode'] = 'predict'

    # 初始化预测器并开始预测
    predictor = PosePredictor(overrides=overrides)
    predictor.predict_cli()


if __name__ == "__main__":
    main()
