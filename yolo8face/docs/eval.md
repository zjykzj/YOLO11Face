
# EVAL

## YOLOv8

### YOLOv8n

```shell
# python3 yolo8face_eval.py --model yolov8n_widerface.pt --data ./yolo8face/cfg/datasets/widerface.yaml --imgsz 640 --device 0
args: Namespace(data='./yolo8face/cfg/datasets/widerface.yaml', device=[0], model='yolov8n_widerface.pt', source=None) - unknown: ['--imgsz', '640']
Ultralytics YOLOv8.2.103 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)
Model summary (fused): 168 layers, 3,005,843 parameters, 0 gradients, 8.1 GFLOPs
val: Scanning /data/zj/datasets/widerface/labels/val.cache... 3226 images, 0 backgrounds, 1 corrupt: 100%|██████████| 3226/32
val: WARNING ⚠️ /data/zj/datasets/widerface/images/val/21_Festival_Festival_21_604.jpg: 1 duplicate labels removed
val: WARNING ⚠️ /data/zj/datasets/widerface/images/val/39_Ice_Skating_iceskiing_39_583.jpg: ignoring corrupt image/label: non-normalized or out of bounds coordinates [      1.002]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 202/202 [00:18<00:
                   all       3225      39675      0.841      0.589      0.669      0.368
Speed: 0.2ms preprocess, 1.2ms inference, 0.0ms loss, 0.6ms postprocess per image
Results saved to /data/zj/YOLO8Face/runs/detect/val3
```

### YOLOv8s

```shell
# python3 yolo8face_eval.py --model yolov8s_widerface.pt --data ./yolo8face/cfg/datasets/widerface.yaml --imgsz 640 --device 0
args: Namespace(data='./yolo8face/cfg/datasets/widerface.yaml', device=[0], model='yolov8s_widerface.pt', source=None) - unknown: ['--imgsz', '640']
Ultralytics YOLOv8.2.103 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)
Model summary (fused): 168 layers, 11,125,971 parameters, 0 gradients, 28.4 GFLOPs
val: Scanning /data/zj/datasets/widerface/labels/val.cache... 3226 images, 0 backgrounds, 1 corrupt: 100%|██████████| 3226/32
val: WARNING ⚠️ /data/zj/datasets/widerface/images/val/21_Festival_Festival_21_604.jpg: 1 duplicate labels removed
val: WARNING ⚠️ /data/zj/datasets/widerface/images/val/39_Ice_Skating_iceskiing_39_583.jpg: ignoring corrupt image/label: non-normalized or out of bounds coordinates [      1.002]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 202/202 [00:21<00:
                   all       3225      39675      0.867      0.629      0.717      0.402
Speed: 0.2ms preprocess, 2.2ms inference, 0.0ms loss, 0.7ms postprocess per image
Results saved to /data/zj/YOLO8Face/runs/detect/val4
```

## YOLOv5

### YOLOv5n

```shell
# python3 yolo8face_eval.py --model yolov5nu_widerface.pt --data ./yolo8face/cfg/datasets/widerface.yaml --imgsz 640 --device 0
args: Namespace(data='./yolo8face/cfg/datasets/widerface.yaml', device=[0], model='yolov5nu_widerface.pt', source=None) - unknown: ['--imgsz', '640']
Ultralytics YOLOv8.2.103 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)
YOLOv5n summary (fused): 193 layers, 2,503,139 parameters, 0 gradients, 7.1 GFLOPs
val: Scanning /data/zj/datasets/widerface/labels/val.cache... 3226 images, 0 backgrounds, 1 corrupt: 100%|██████████| 3226/32
val: WARNING ⚠️ /data/zj/datasets/widerface/images/val/21_Festival_Festival_21_604.jpg: 1 duplicate labels removed
val: WARNING ⚠️ /data/zj/datasets/widerface/images/val/39_Ice_Skating_iceskiing_39_583.jpg: ignoring corrupt image/label: non-normalized or out of bounds coordinates [      1.002]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 202/202 [00:18<00:
                   all       3225      39675       0.84       0.58       0.66      0.363
Speed: 0.2ms preprocess, 1.2ms inference, 0.0ms loss, 0.6ms postprocess per image
Results saved to /data/zj/YOLO8Face/runs/detect/val
```

### YOLOv5s

```shell
# python3 yolo8face_eval.py --model yolov5su_widerface.pt --data ./yolo8face/cfg/datasets/widerface.yaml --imgsz 640 --device 0
args: Namespace(data='./yolo8face/cfg/datasets/widerface.yaml', device=[0], model='yolov5su_widerface.pt', source=None) - unknown: ['--imgsz', '640']
Ultralytics YOLOv8.2.103 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)
YOLOv5s summary (fused): 193 layers, 9,111,923 parameters, 0 gradients, 23.8 GFLOPs
val: Scanning /data/zj/datasets/widerface/labels/val.cache... 3226 images, 0 backgrounds, 1 corrupt: 100%|██████████| 3226/32
val: WARNING ⚠️ /data/zj/datasets/widerface/images/val/21_Festival_Festival_21_604.jpg: 1 duplicate labels removed
val: WARNING ⚠️ /data/zj/datasets/widerface/images/val/39_Ice_Skating_iceskiing_39_583.jpg: ignoring corrupt image/label: non-normalized or out of bounds coordinates [      1.002]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 202/202 [00:21<00:
                   all       3225      39675      0.863       0.63      0.713      0.398
Speed: 0.2ms preprocess, 2.1ms inference, 0.0ms loss, 0.8ms postprocess per image
Results saved to /data/zj/YOLO8Face/runs/detect/val2
```