# -*- coding: utf-8 -*-

"""
@Time    : 2025/1/19 16:32
@File    : eval.py
@Author  : zj
@Description: 
"""

import argparse

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
