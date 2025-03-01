
# YOLO5Face

## yolov5su

```shell
# python3 detect_eval.py --model yolov5nu_widerface.pt --data ./yolo11face/cfg/datasets/widerface.yaml --imgsz 640 --device 0
args: Namespace(data='./yolo11face/cfg/datasets/widerface.yaml', device=[0], model='yolov5nu_widerface.pt', source=None) - unknown: ['--imgsz', '640']
Ultralytics 8.3.75 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)
YOLOv5n summary (fused): 193 layers, 2,503,139 parameters, 0 gradients, 7.1 GFLOPs
val: Scanning /data/zj/datasets/widerface/labels/val.cache... 3226 images, 0 backgrounds, 1 corrupt: 100%|██████████| 3226/3226 [00:00<?, ?it/s]
val: WARNING ⚠️ /data/zj/datasets/widerface/images/val/21_Festival_Festival_21_604.jpg: 1 duplicate labels removed
val: WARNING ⚠️ /data/zj/datasets/widerface/images/val/39_Ice_Skating_iceskiing_39_583.jpg: ignoring corrupt image/label: non-normalized or out of bounds coordinates [      1.002]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 202/202 [00:22<00:00,  9.14it/s]
                   all       3225      39675      0.856      0.578      0.666      0.369
Speed: 0.2ms preprocess, 1.3ms inference, 0.0ms loss, 1.5ms postprocess per image
Results saved to /data/zj/YOLO11Face/runs/detect/val3
```

## yolov5nu

```shell
# python3 detect_eval.py --model yolov5su_widerface.pt --data ./yolo11face/cfg/datasets/widerface.yaml --imgsz 640 --device 0
args: Namespace(data='./yolo11face/cfg/datasets/widerface.yaml', device=[0], model='yolov5su_widerface.pt', source=None) - unknown: ['--imgsz', '640']
Ultralytics 8.3.75 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)
YOLOv5s summary (fused): 193 layers, 9,111,923 parameters, 0 gradients, 23.8 GFLOPs
val: Scanning /data/zj/datasets/widerface/labels/val.cache... 3226 images, 0 backgrounds, 1 corrupt: 100%|██████████| 3226/3226 [00:00<?, ?it/s]
val: WARNING ⚠️ /data/zj/datasets/widerface/images/val/21_Festival_Festival_21_604.jpg: 1 duplicate labels removed
val: WARNING ⚠️ /data/zj/datasets/widerface/images/val/39_Ice_Skating_iceskiing_39_583.jpg: ignoring corrupt image/label: non-normalized or out of bounds coordinates [      1.002]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 202/202 [00:24<00:00,  8.25it/s]
                   all       3225      39675      0.877      0.624      0.717      0.405
Speed: 0.2ms preprocess, 2.1ms inference, 0.0ms loss, 1.7ms postprocess per image
```