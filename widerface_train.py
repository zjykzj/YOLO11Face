# -*- coding: utf-8 -*-

"""
@date: 2025/1/19 下午8:58
@file: widerface_train.py
@author: zj
@description: 
"""

from ultralytics.models.yolo.detect import DetectionTrainer

args = dict(model="yolov8n.pt", data="widerface.yaml", device=0)
trainer = DetectionTrainer(overrides=args)
trainer.train()