
# YOLO8Face

### YOLOv8s-pose

```shell
# python3 pose_eval.py --model yolov8s-pose_widerface.pt --data ./yolo11face/cfg/datasets/widerface-landmarks.yaml --imgsz 640 --device 0
args: Namespace(data='./yolo11face/cfg/datasets/widerface-landmarks.yaml', device=[0], model='yolov8s-pose_widerface.pt', source=None) - unknown: ['--imgsz', '640']
Ultralytics 8.3.75 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)
YOLOv8s-pose summary (fused): 187 layers, 11,413,344 parameters, 0 gradients, 29.4 GFLOPs
val: Scanning /data/zj/datasets/widerface-landmarks/labels/val.cache... 2576 images, 0 backgrounds, 0 corrupt: 100%|██████████| 2576/2576 [00:00<?, ?it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 161/161 [00:52<00:00,  3.08it/s]
                   all       2576      29560      0.855      0.599      0.683      0.392       0.78      0.401      0.442      0.424
Speed: 0.3ms preprocess, 3.4ms inference, 0.0ms loss, 1.2ms postprocess per image
Results saved to /data/zj/YOLO11Face/runs/detect/val4
```

### YOLOv8n-pose

```shell
# python3 pose_eval.py --model yolov8n-pose_widerface.pt --data ./yolo11face/cfg/datasets/widerface-landmarks.yaml --imgsz 640 --device 0
args: Namespace(data='./yolo11face/cfg/datasets/widerface-landmarks.yaml', device=[0], model='yolov8n-pose_widerface.pt', source=None) - unknown: ['--imgsz', '640']
Ultralytics 8.3.75 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)
YOLOv8n-pose summary (fused): 187 layers, 3,078,128 parameters, 0 gradients, 8.3 GFLOPs
val: Scanning /data/zj/datasets/widerface-landmarks/labels/val.cache... 2576 images, 0 backgrounds, 0 corrupt: 100%|██████████| 2576/2576 [00:00<?, ?it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 161/161 [00:48<00:00,  3.31it/s]
                   all       2576      29560      0.845      0.552      0.636       0.36      0.776      0.394      0.429      0.403
Speed: 0.3ms preprocess, 2.2ms inference, 0.0ms loss, 1.3ms postprocess per image
Results saved to /data/zj/YOLO11Face/runs/detect/val
```