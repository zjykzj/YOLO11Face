
# EVAL

## yolov8n-pose

```shell
# python3 pose_eval.py --model yolov8n-pose_widerface.pt --data ./yolo8face/cfg/datasets/widerface-landmarks.yaml --device 0
args: Namespace(data='./yolo8face/cfg/datasets/widerface-landmarks.yaml', device=[7], model='yolov8n-pose_widerface.pt', source=None) - unknown: []
Ultralytics YOLOv8.2.103 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:7 (NVIDIA GeForce RTX 3090, 24268MiB)
YOLOv8n-pose summary (fused): 187 layers, 3,078,128 parameters, 0 gradients, 8.3 GFLOPs
val: Scanning /data/zj/datasets/widerface-landmarks/labels/val.cache... 2576 images, 0 backgrounds, 0 corrupt: 100%|██████████| 2576/2576 [00:00<?, ?it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 161/161 [00:51<00:00,  3.15it/s]
                   all       2576      29560      0.843      0.552      0.632      0.353       0.76      0.393      0.423      0.396
Speed: 0.3ms preprocess, 2.8ms inference, 0.0ms loss, 1.1ms postprocess per image
Results saved to /data/zj/YOLO8Face/runs/detect/val2
```

## yolov8s-pose

```shell
# python3 pose_eval.py --model yolov8s-pose_widerface.pt --data ./yolo8face/cfg/datasets/widerface-landmarks.yaml --device 0
args: Namespace(data='./yolo8face/cfg/datasets/widerface-landmarks.yaml', device=[0], model='yolov8s-pose_widerface.pt', source=None) - unknown: []
Ultralytics YOLOv8.2.103 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)
YOLOv8s-pose summary (fused): 187 layers, 11,413,344 parameters, 0 gradients, 29.4 GFLOPs
val: Scanning /data/zj/datasets/widerface-landmarks/labels/val.cache... 2576 images, 0 backgrounds, 0 corrupt: 100%|██████████| 2576/2576 [00:00<?, ?it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 161/161 [00:53<00:00,  3.03it/s]
                   all       2576      29560      0.858      0.599      0.681      0.385      0.783      0.398      0.436      0.417
Speed: 0.2ms preprocess, 3.0ms inference, 0.0ms loss, 1.3ms postprocess per image
Results saved to /data/zj/YOLO8Face/runs/detect/val3
```