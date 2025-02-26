
# YOLO11Face

## YOLO11s-pose

```shell
# python3 pose_eval.py --model yolo11s-pose_widerface.pt --data ./yolo11face/cfg/datasets/widerface-landmarks.yaml --imgsz 640 --device 0
args: Namespace(data='./yolo11face/cfg/datasets/widerface-landmarks.yaml', device=[0], model='yolo11s-pose_widerface.pt', source=None) - unknown: ['--imgsz', '640']
Ultralytics 8.3.75 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)
YOLO11s-pose summary (fused): 257 layers, 9,700,560 parameters, 0 gradients, 22.3 GFLOPs
val: Scanning /data/zj/datasets/widerface-landmarks/labels/val.cache... 2576 images, 0 backgrounds, 0 corrupt: 100%|██████████| 2576/2576 [00:00<?, ?it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 161/161 [00:52<00:00,  3.04it/s]
                   all       2576      29560      0.858      0.609      0.692      0.399      0.785      0.399      0.444      0.427
Speed: 0.3ms preprocess, 3.5ms inference, 0.0ms loss, 1.3ms postprocess per image
Results saved to /data/zj/YOLO11Face/runs/detect/val5
```

## YOLO11n-pose

```shell
# python3 pose_eval.py --model yolo11n-pose_widerface.pt --data ./yolo11face/cfg/datasets/widerface-landmarks.yaml --imgsz 640 --device 0
args: Namespace(data='./yolo11face/cfg/datasets/widerface-landmarks.yaml', device=[0], model='yolo11n-pose_widerface.pt', source=None) - unknown: ['--imgsz', '640']
Ultralytics 8.3.75 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)
YOLO11n-pose summary (fused): 257 layers, 2,654,632 parameters, 0 gradients, 6.6 GFLOPs
val: Scanning /data/zj/datasets/widerface-landmarks/labels/val.cache... 2576 images, 0 backgrounds, 0 corrupt: 100%|██████████| 2576/2576 [00:00<?, ?it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 161/161 [00:48<00:00,  3.29it/s]
                   all       2576      29560      0.843      0.556       0.64      0.362      0.783      0.393      0.429      0.405
Speed: 0.3ms preprocess, 2.9ms inference, 0.0ms loss, 1.5ms postprocess per image
Results saved to /data/zj/YOLO11Face/runs/detect/val3
```