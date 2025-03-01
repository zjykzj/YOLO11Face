
# YOLO11

## yolo11s

```shell
# python3 detect_eval.py --model yolo11s_widerface.pt --data ./yolo11face/cfg/datasets/widerface.yaml --imgsz 640 --device 0
args: Namespace(data='./yolo11face/cfg/datasets/widerface.yaml', device=[0], model='yolo11s_widerface.pt', source=None) - unknown: ['--imgsz', '640']
Ultralytics 8.3.75 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)
YOLO11s summary (fused): 238 layers, 9,413,187 parameters, 0 gradients, 21.3 GFLOPs
val: Scanning /data/zj/datasets/widerface/labels/val.cache... 3226 images, 0 backgrounds, 1 corrupt: 100%|██████████| 3226/3226 [00:00<?, ?it/s]
val: WARNING ⚠️ /data/zj/datasets/widerface/images/val/21_Festival_Festival_21_604.jpg: 1 duplicate labels removed
val: WARNING ⚠️ /data/zj/datasets/widerface/images/val/39_Ice_Skating_iceskiing_39_583.jpg: ignoring corrupt image/label: non-normalized or out of bounds coordinates [      1.002]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 202/202 [00:24<00:00,  8.24it/s]
                   all       3225      39675      0.869      0.635      0.725      0.411
Speed: 0.2ms preprocess, 2.7ms inference, 0.0ms loss, 1.3ms postprocess per image
Results saved to /data/zj/YOLO11Face/runs/detect/val
```