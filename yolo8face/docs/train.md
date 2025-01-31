
# TRAIN

## YOLOv8

### YOLOv8n

```shell
# python3 yolo8face_train.py --model yolov8n.pt --data ./yolo8face/cfg/datasets/widerface.yaml --epochs 100 --imgsz 640 --device 0
...
...
...
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     99/100      18.3G      1.268     0.5815     0.9499        204        640: 100%|██████████| 805/805 [01:30<00:00,  8.92it
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 101/101 [00:11<00:
                   all       3225      39675      0.842      0.586      0.668      0.366

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    100/100      20.5G      1.269     0.5813     0.9481         63        640: 100%|██████████| 805/805 [01:32<00:00,  8.74it
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 101/101 [00:11<00:
                   all       3225      39675      0.842      0.587      0.668      0.367

100 epochs completed in 2.883 hours.
Optimizer stripped from /data/zj/YOLO8Face/runs/detect/train3/weights/last.pt, 6.2MB
Optimizer stripped from /data/zj/YOLO8Face/runs/detect/train3/weights/best.pt, 6.2MB

Validating /data/zj/YOLO8Face/runs/detect/train3/weights/best.pt...
Ultralytics YOLOv8.2.103 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:6 (NVIDIA GeForce RTX 3090, 24268MiB)
Model summary (fused): 168 layers, 3,005,843 parameters, 0 gradients, 8.1 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 101/101 [00:16<00:
                   all       3225      39675      0.843      0.586      0.668      0.366
Speed: 0.2ms preprocess, 0.5ms inference, 0.0ms loss, 0.5ms postprocess per image
Results saved to /data/zj/YOLO8Face/runs/detect/train3
```

### YOLOv8s

```shell
# python3 yolo8face_train.py --model yolov8s.pt --data ./yolo8face/cfg/datasets/widerface.yaml --epochs 100 --imgsz 640 --device 0
...
...
...
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     99/100      19.9G      1.108     0.4856     0.8968        204        640: 100%|██████████| 805/805 [01:32<00:00,  8.67it
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 101/101 [00:12<00:
                   all       3225      39675      0.865       0.63      0.716        0.4

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    100/100      21.9G       1.11     0.4852     0.8955         63        640: 100%|██████████| 805/805 [01:32<00:00,  8.72it
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 101/101 [00:12<00:
                   all       3225      39675      0.866      0.629      0.716      0.401

100 epochs completed in 3.067 hours.
Optimizer stripped from /data/zj/YOLO8Face/runs/detect/train4/weights/last.pt, 22.5MB
Optimizer stripped from /data/zj/YOLO8Face/runs/detect/train4/weights/best.pt, 22.5MB

Validating /data/zj/YOLO8Face/runs/detect/train4/weights/best.pt...
Ultralytics YOLOv8.2.103 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:5 (NVIDIA GeForce RTX 3090, 24268MiB)
Model summary (fused): 168 layers, 11,125,971 parameters, 0 gradients, 28.4 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 101/101 [00:17<00:
                   all       3225      39675      0.865       0.63      0.716      0.401
Speed: 0.2ms preprocess, 1.0ms inference, 0.0ms loss, 0.5ms postprocess per image
Results saved to /data/zj/YOLO8Face/runs/detect/train4
```

## YOLOv5

### YOLOv5n

```shell
# python3 yolo8face_train.py --model yolov5nu.pt --data ./yolo8face/cfg/datasets/widerface.yaml --epochs 100 --imgsz 640 --device 0
...
...
...
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     99/100      18.2G      1.301     0.6122     0.9408        204        640: 100%|██████████| 805/805 [01:36<00:00,  8.30it
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 101/101 [00:11<00:
                   all       3225      39675      0.839      0.581       0.66       0.36

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    100/100      20.2G      1.303      0.611     0.9397         63        640: 100%|██████████| 805/805 [01:36<00:00,  8.34it
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 101/101 [00:11<00:
                   all       3225      39675       0.84       0.58      0.661      0.361

100 epochs completed in 2.987 hours.
Optimizer stripped from /data/zj/YOLO8Face/runs/detect/train/weights/last.pt, 5.3MB
Optimizer stripped from /data/zj/YOLO8Face/runs/detect/train/weights/best.pt, 5.3MB

Validating /data/zj/YOLO8Face/runs/detect/train/weights/best.pt...
Ultralytics YOLOv8.2.103 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:7 (NVIDIA GeForce RTX 3090, 24268MiB)
YOLOv5n summary (fused): 193 layers, 2,503,139 parameters, 0 gradients, 7.1 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 101/101 [00:15<00:
                   all       3225      39675       0.84       0.58       0.66      0.361
Speed: 0.1ms preprocess, 0.5ms inference, 0.0ms loss, 0.5ms postprocess per image
Results saved to /data/zj/YOLO8Face/runs/detect/train
```

### YOLOv5s

```shell
# python3 yolo8face_train.py --model yolov5su.pt --data ./yolo8face/cfg/datasets/widerface.yaml --epochs 100 --imgsz 640 --device 0
...
...
...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     99/100      19.8G      1.159     0.5156     0.9039        204        640: 100%|██████████| 805/805 [01:38<00:00,  8.20it
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 101/101 [00:12<00:
                   all       3225      39675      0.865      0.627      0.712      0.397

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    100/100      21.8G      1.159     0.5136      0.902         63        640: 100%|██████████| 805/805 [01:38<00:00,  8.19it
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 101/101 [00:12<00:
                   all       3225      39675      0.865      0.627      0.712      0.397

100 epochs completed in 3.130 hours.
Optimizer stripped from /data/zj/YOLO8Face/runs/detect/train2/weights/last.pt, 18.5MB
Optimizer stripped from /data/zj/YOLO8Face/runs/detect/train2/weights/best.pt, 18.5MB

Validating /data/zj/YOLO8Face/runs/detect/train2/weights/best.pt...
Ultralytics YOLOv8.2.103 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:7 (NVIDIA GeForce RTX 3090, 24268MiB)
YOLOv5s summary (fused): 193 layers, 9,111,923 parameters, 0 gradients, 23.8 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 101/101 [00:16<00:
                   all       3225      39675      0.865      0.627      0.712      0.397
Speed: 0.2ms preprocess, 1.0ms inference, 0.0ms loss, 0.5ms postprocess per image
Results saved to /data/zj/YOLO8Face/runs/detect/train2
```