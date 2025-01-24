# -*- coding: utf-8 -*-

"""
@Time    : 2025/1/19 16:51
@File    : widerface_predict.py
@Author  : zj
@Description: 
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
