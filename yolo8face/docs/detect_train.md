
# TRAIN

## YOLOv8

### YOLOv8n

* E100 + I640

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

* E100 + I800

```shell
# python3 yolo8face_train.py --model yolov5nu.pt --data ./yolo8face/cfg/datasets/widerface.yaml --epochs 300 --imgsz 800 --batch 8 --device 7
args: Namespace(data='./yolo8face/cfg/datasets/widerface.yaml', device=[7], model='yolov5nu.pt', source=None) - unknown: ['--epochs', '300', '--imgsz', '800', '--batch', '8']
Ultralytics YOLOv8.2.103 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:7 (NVIDIA GeForce RTX 3090, 24268MiB)
WARNING ⚠️ Upgrade to torch>=2.0.0 for deterministic training.
engine/trainer: task=detect, mode=train, model=yolov5nu.pt, data=./yolo8face/cfg/datasets/widerface.yaml, epochs=300, time=None, patience=100, batch=8, imgsz=800, save=True, save_period=-1, cache=False, device=[7], workers=8, project=None, name=train8, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=True, opset=None, workspace=4, nms=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, label_smoothing=0.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, copy_paste=0.0, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=/data/zj/YOLO8Face/runs/detect/train8
Overriding model.yaml nc=80 with nc=1
...
...
...
YOLOv5n summary: 262 layers, 2,508,659 parameters, 2,508,643 gradients, 7.2 GFLOPs

Transferred 391/427 items from pretrained weights
Freezing layer 'model.24.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks with YOLOv8n...
AMP: checks passed ✅
train: Scanning /data/zj/datasets/widerface/labels/train.cache... 12876 images, 0 backgrounds, 1 corrupt: 100%|██████████| 12
train: WARNING ⚠️ /data/zj/datasets/widerface/images/train/2_Demonstration_Protesters_2_231.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /data/zj/datasets/widerface/images/train/37_Soccer_Soccer_37_851.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /data/zj/datasets/widerface/images/train/54_Rescue_rescuepeople_54_29.jpg: ignoring corrupt image/label: non-normalized or out of bounds coordinates [     1.0254]
train: WARNING ⚠️ /data/zj/datasets/widerface/images/train/7_Cheering_Cheering_7_17.jpg: 1 duplicate labels removed
val: Scanning /data/zj/datasets/widerface/labels/val.cache... 3226 images, 0 backgrounds, 1 corrupt: 100%|██████████| 3226/32
val: WARNING ⚠️ /data/zj/datasets/widerface/images/val/21_Festival_Festival_21_604.jpg: 1 duplicate labels removed
val: WARNING ⚠️ /data/zj/datasets/widerface/images/val/39_Ice_Skating_iceskiing_39_583.jpg: ignoring corrupt image/label: non-normalized or out of bounds coordinates [      1.002]
Plotting labels to /data/zj/YOLO8Face/runs/detect/train8/labels.jpg...
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically...
optimizer: SGD(lr=0.01, momentum=0.9) with parameter groups 69 weight(decay=0.0), 76 weight(decay=0.0005), 75 bias(decay=0.0)
Image sizes 800 train, 800 val
Using 8 dataloader workers
Logging results to /data/zj/YOLO8Face/runs/detect/train8
Starting training for 300 epochs...
...
...
...
```

### YOLOv8s

* E100 + I640

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

* E300 + I800

```shell
# python3 yolo8face_train.py --model yolov8s.pt --data ./yolo8face/cfg/datasets/widerface.yaml --epochs 300 --imgsz 800 --device 0 --batch 8
...
...
...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    299/300      13.9G      0.915     0.4065     0.8607         58        800: 100%|██████████| 1610/1610 [02:44<00:00,  9.80
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 202/202 [00:16<00:
                   all       3225      39675      0.882      0.677      0.766      0.431

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    300/300      18.4G     0.9144      0.408     0.8605         30        800: 100%|██████████| 1610/1610 [02:44<00:00,  9.79
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 202/202 [00:16<00:
                   all       3225      39675      0.881      0.677      0.766      0.431

300 epochs completed in 15.311 hours.
Optimizer stripped from /data/zj/YOLO8Face/runs/detect/train2/weights/last.pt, 22.5MB
Optimizer stripped from /data/zj/YOLO8Face/runs/detect/train2/weights/best.pt, 22.5MB

Validating /data/zj/YOLO8Face/runs/detect/train2/weights/best.pt...
Ultralytics YOLOv8.2.103 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:6 (NVIDIA GeForce RTX 3090, 24268MiB)
Model summary (fused): 168 layers, 11,125,971 parameters, 0 gradients, 28.4 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 202/202 [00:20<00:
                   all       3225      39675      0.881       0.68      0.766      0.434
Speed: 0.2ms preprocess, 1.8ms inference, 0.0ms loss, 0.6ms postprocess per image
Results saved to /data/zj/YOLO8Face/runs/detect/train2
```

## YOLOv5

### YOLOv5n

* E100 + I640

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

* E300 + I800

```shell
# python3 yolo8face_train.py --model yolov5nu.pt --data ./yolo8face/cfg/datasets/widerface.yaml --epochs 300 --imgsz 800 --device 0 --batch 8
...
...
...
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    299/300      12.5G      1.214     0.5665     0.9363         58        800: 100%|██████████| 1610/1610 [02:56<00:00,  9.10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 202/202 [00:14<00:
                   all       3225      39675      0.868      0.632       0.72        0.4

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    300/300        17G      1.217     0.5704     0.9367         30        800: 100%|██████████| 1610/1610 [02:56<00:00,  9.14
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 202/202 [00:14<00:
                   all       3225      39675      0.868      0.632       0.72      0.401

300 epochs completed in 16.288 hours.
Optimizer stripped from /data/zj/YOLO8Face/runs/detect/train3/weights/last.pt, 5.3MB
Optimizer stripped from /data/zj/YOLO8Face/runs/detect/train3/weights/best.pt, 5.3MB

Validating /data/zj/YOLO8Face/runs/detect/train3/weights/best.pt...
Ultralytics YOLOv8.2.103 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:5 (NVIDIA GeForce RTX 3090, 24268MiB)
YOLOv5n summary (fused): 193 layers, 2,503,139 parameters, 0 gradients, 7.1 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 202/202 [00:17<00:
                   all       3225      39675      0.867      0.632       0.72      0.401
Speed: 0.2ms preprocess, 0.9ms inference, 0.0ms loss, 0.5ms postprocess per image
Results saved to /data/zj/YOLO8Face/runs/detect/train3
```

### YOLOv5s

* E100 + I640

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

* E300 + I800

```shell
# python3 yolo8face_train.py --model yolov5su.pt --data ./yolo8face/cfg/datasets/widerface.yaml --epochs 300 --imgsz 800 --device 0 --batch 8
...
...
...
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    299/300      13.8G     0.9745     0.4388     0.8703         58        800: 100%|██████████| 1610/1610 [02:55<00:00,  9.18
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 202/202 [00:15<00:
                   all       3225      39675       0.88      0.675      0.765       0.43

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    300/300      18.4G     0.9732     0.4408     0.8701         30        800: 100%|██████████| 1610/1610 [02:52<00:00,  9.32
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 202/202 [00:15<00:
                   all       3225      39675      0.879      0.675      0.765       0.43

300 epochs completed in 16.396 hours.
Optimizer stripped from /data/zj/YOLO8Face/runs/detect/train/weights/last.pt, 18.5MB
Optimizer stripped from /data/zj/YOLO8Face/runs/detect/train/weights/best.pt, 18.5MB

Validating /data/zj/YOLO8Face/runs/detect/train/weights/best.pt...
Ultralytics YOLOv8.2.103 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:7 (NVIDIA GeForce RTX 3090, 24268MiB)
YOLOv5s summary (fused): 193 layers, 9,111,923 parameters, 0 gradients, 23.8 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 202/202 [00:19<00:
                   all       3225      39675      0.882      0.674      0.764      0.431
Speed: 0.2ms preprocess, 1.5ms inference, 0.0ms loss, 0.6ms postprocess per image
Results saved to /data/zj/YOLO8Face/runs/detect/train
```