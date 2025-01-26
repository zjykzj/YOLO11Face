
# TRAIN

## YOLOv8

### YOLOv8n

```shell
root@a988b27f02e9:/data/zj/YOLO8Face# python3 yolo8face_train.py --model yolov8n.pt --data ./yolo8face/cfg/datasets/widerface.yaml --device 0
args: Namespace(data='./yolo8face/cfg/datasets/widerface.yaml', device=[7], model='yolov8n.pt', source='./yolo8face/assets') - unknown: []
Ultralytics YOLOv8.2.103 🚀 Python-3.8.10 torch-1.14.0a0+410ce96 CUDA:7 (NVIDIA GeForce RTX 3090, 24268MiB)
WARNING ⚠️ Upgrade to torch>=2.0.0 for deterministic training.
engine/trainer: task=detect, mode=train, model=yolov8n.pt, data=./yolo8face/cfg/datasets/widerface.yaml, epochs=100, time=None, patience=100, batch=16, imgsz=640, save=True, save_period=-1, cache=False, dev$ce=[7], workers=8, project=None, name=train6, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=F$lse, amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det$300, half=False, dnn=False, plots=True, source=./yolo8face/assets, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=F$lse, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dyn$mic=False, simplify=True, opset=None, workspace=4, nms=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, $ose=12.0, kobj=1.0, label_smoothing=0.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, 
copy_paste=0.0, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=/data/zj/YOLO8Face/runs/detect/train6
TensorBoard: Start with 'tensorboard --logdir /data/zj/YOLO8Face/runs/detect/train6', view at http://localhost:6006/
Overriding model.yaml nc=80 with nc=1

                   from  n    params  module                                       arguments                     
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]                 
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]                
  2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]             
  3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]                
  4                  -1  2     49664  ultralytics.nn.modules.block.C2f             [64, 64, 2, True]             
  5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  6                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]           
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              
  8                  -1  1    460288  ultralytics.nn.modules.block.C2f             [256, 256, 1, True]           
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]                 
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 12                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]                 
 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 15                  -1  1     37248  ultralytics.nn.modules.block.C2f             [192, 64, 1]                  
 16                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 18                  -1  1    123648  ultralytics.nn.modules.block.C2f             [192, 128, 1]                 
 19                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 21                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]                 
 22        [15, 18, 21]  1    751507  ultralytics.nn.modules.head.Detect           [1, [64, 128, 256]]           
Model summary: 225 layers, 3,011,043 parameters, 3,011,027 gradients

Transferred 319/355 items from pretrained weights
Freezing layer 'model.22.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks with YOLOv8n...
AMP: checks passed ✅
train: Scanning /data/zj/datasets/widerface/labels/train.cache... 12876 images, 0 backgrounds, 1 corrupt: 100%|██████████| 12876/12876 [00:00<?, ?it/s]
train: WARNING ⚠️ /data/zj/datasets/widerface/images/train/2_Demonstration_Protesters_2_231.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /data/zj/datasets/widerface/images/train/37_Soccer_Soccer_37_851.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /data/zj/datasets/widerface/images/train/54_Rescue_rescuepeople_54_29.jpg: ignoring corrupt image/label: non-normalized or out of bounds coordinates [     1.0254]
train: WARNING ⚠️ /data/zj/datasets/widerface/images/train/7_Cheering_Cheering_7_17.jpg: 1 duplicate labels removed
val: Scanning /data/zj/datasets/widerface/labels/val.cache... 3226 images, 0 backgrounds, 1 corrupt: 100%|██████████| 3226/3226 [00:00<?, ?it/s]
val: WARNING ⚠️ /data/zj/datasets/widerface/images/val/21_Festival_Festival_21_604.jpg: 1 duplicate labels removed
val: WARNING ⚠️ /data/zj/datasets/widerface/images/val/39_Ice_Skating_iceskiing_39_583.jpg: ignoring corrupt image/label: non-normalized or out of bounds coordinates [      1.002]
Plotting labels to /data/zj/YOLO8Face/runs/detect/train6/labels.jpg... 
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: SGD(lr=0.01, momentum=0.9) with parameter groups 57 weight(decay=0.0), 64 weight(decay=0.0005), 63 bias(decay=0.0)
TensorBoard: model graph visualization added ✅
Image sizes 640 train, 640 val
Using 8 dataloader workers
Logging results to /data/zj/YOLO8Face/runs/detect/train6
Starting training for 100 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/100      15.5G      1.893      1.648      1.227        308        640: 100%|██████████| 805/805 [01:22<00:00,  9.75it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 101/101 [00:15<00:00,  6.37it/s]
                   all       3225      39675      0.725       0.45      0.499      0.245
...
...
...
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     99/100      18.2G      1.267      0.582     0.9497        204        640: 100%|██████████| 805/805 [01:08<00:00, 11.67it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 101/101 [00:10<00:00,  9.54it/s]
                   all       3225      39675      0.844      0.586      0.668      0.366
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    100/100      20.5G      1.269     0.5829     0.9482         63        640: 100%|██████████| 805/805 [01:09<00:00, 11.52it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 101/101 [00:10<00:00,  9.53it/s]
                   all       3225      39675      0.843      0.586      0.668      0.366

100 epochs completed in 2.246 hours.
Optimizer stripped from /data/zj/YOLO8Face/runs/detect/train6/weights/last.pt, 6.2MB
Optimizer stripped from /data/zj/YOLO8Face/runs/detect/train6/weights/best.pt, 6.2MB

Validating /data/zj/YOLO8Face/runs/detect/train6/weights/best.pt...
Ultralytics YOLOv8.2.103 🚀 Python-3.8.10 torch-1.14.0a0+410ce96 CUDA:7 (NVIDIA GeForce RTX 3090, 24268MiB)
Model summary (fused): 168 layers, 3,005,843 parameters, 0 gradients
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 101/101 [00:18<00:00,  5.36it/s]
                   all       3225      39675      0.843      0.586      0.668      0.366
Speed: 0.1ms preprocess, 0.4ms inference, 0.0ms loss, 0.5ms postprocess per image
Results saved to /data/zj/YOLO8Face/runs/detect/train6
```

### YOLOv8s

```shell
root@a988b27f02e9:/data/zj/YOLO8Face# python3 yolo8face_train.py --model yolov8s.pt --data ./yolo8face/cfg/datasets/widerface.yaml --device 6
args: Namespace(data='./yolo8face/cfg/datasets/widerface.yaml', device=[6], model='yolov8s.pt', source='./yolo8face/assets') - unknown: [] 
Ultralytics YOLOv8.2.103 🚀 Python-3.8.10 torch-1.14.0a0+410ce96 CUDA:6 (NVIDIA GeForce RTX 3090, 24268MiB)
WARNING ⚠️ Upgrade to torch>=2.0.0 for deterministic training.
engine/trainer: task=detect, mode=train, model=yolov8s.pt, data=./yolo8face/cfg/datasets/widerface.yaml, epochs=100, time=None, patience=100, batch=16, imgsz=640, save=True, save_period=-1, cache=False, device=[6], workers=8, project=None, name=train7, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=./yolo8face/assets, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=True, opset=None, workspace=4, nms=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, label_smoothing=0.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, copy_paste=0.0, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=/data/zj/YOLO8Face/runs/detect/train7
TensorBoard: Start with 'tensorboard --logdir /data/zj/YOLO8Face/runs/detect/train7', view at http://localhost:6006/
Overriding model.yaml nc=80 with nc=1

                   from  n    params  module                                       arguments  
  0                  -1  1       928  ultralytics.nn.modules.conv.Conv             [3, 32, 3, 2]                   
  1                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]                  
  2                  -1  1     29056  ultralytics.nn.modules.block.C2f             [64, 64, 1, True]               
  3                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]                 
  4                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]             
  5                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]                
  6                  -1  2    788480  ultralytics.nn.modules.block.C2f             [256, 256, 2, True]             
  7                  -1  1   1180672  ultralytics.nn.modules.conv.Conv             [256, 512, 3, 2]              
  8                  -1  1   1838080  ultralytics.nn.modules.block.C2f             [512, 512, 1, True]           
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]      
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']           
 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                            
 12                  -1  1    591360  ultralytics.nn.modules.block.C2f             [768, 256, 1]                 
 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 15                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]                 
 16                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]               
 18                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]                 
 19                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 21                  -1  1   1969152  ultralytics.nn.modules.block.C2f             [768, 512, 1]     
 22        [15, 18, 21]  1   2116435  ultralytics.nn.modules.head.Detect           [1, [128, 256, 512]]          
Model summary: 225 layers, 11,135,987 parameters, 11,135,971 gradients

Transferred 349/355 items from pretrained weights
Freezing layer 'model.22.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks with YOLOv8n...
AMP: checks passed ✅
train: Scanning /data/zj/datasets/widerface/labels/train.cache... 12876 images, 0 backgrounds, 1 corrupt: 100%|██████████| 12876/12876 [00:00<?, ?it/s]train: WARNING ⚠️ /data/zj/datasets/widerface/images/train/2_Demonstration_Protesters_2_231.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /data/zj/datasets/widerface/images/train/37_Soccer_Soccer_37_851.jpg: 1 duplicate labels removedtrain: WARNING ⚠️ /data/zj/datasets/widerface/images/train/54_Rescue_rescuepeople_54_29.jpg: ignoring corrupt image/label: non-normalized or out of bounds coordinates [     1.0254]train: WARNING ⚠️ /data/zj/datasets/widerface/images/train/7_Cheering_Cheering_7_17.jpg: 1 duplicate labels removedval: Scanning /data/zj/datasets/widerface/labels/val.cache... 3226 images, 0 backgrounds, 1 corrupt: 100%|██████████| 3226/3226 [00:00<?, ?it/s]
val: WARNING ⚠️ /data/zj/datasets/widerface/images/val/21_Festival_Festival_21_604.jpg: 1 duplicate labels removed
val: WARNING ⚠️ /data/zj/datasets/widerface/images/val/39_Ice_Skating_iceskiing_39_583.jpg: ignoring corrupt image/label: non-normalized or out of bounds coordinates [      1.002]
Plotting labels to /data/zj/YOLO8Face/runs/detect/train7/labels.jpg... 
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: SGD(lr=0.01, momentum=0.9) with parameter groups 57 weight(decay=0.0), 64 weight(decay=0.0005), 63 bias(decay=0.0)
TensorBoard: model graph visualization added ✅
Image sizes 640 train, 640 val
Using 8 dataloader workers
Logging results to /data/zj/YOLO8Face/runs/detect/train7
Starting training for 100 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/100      17.1G      1.788      1.253       1.18        308        640: 100%|██████████| 805/805 [01:37<00:00,  8.28it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 101/101 [00:17<00:00,  5.67it/s]
                   all       3225      39675      0.796       0.52      0.595      0.304
...
...
...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     99/100        20G      1.106     0.4844     0.8942        204        640: 100%|██████████| 805/805 [01:22<00:00,  9.75it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 101/101 [00:12<00:00,  8.07it/s]
                   all       3225      39675      0.869       0.63      0.718        0.4

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    100/100        22G      1.108     0.4839     0.8936         63        640: 100%|██████████| 805/805 [01:22<00:00,  9.78it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 101/101 [00:12<00:00,  8.08it/s]
                   all       3225      39675      0.869       0.63      0.718        0.4

100 epochs completed in 2.697 hours.
Optimizer stripped from /data/zj/YOLO8Face/runs/detect/train7/weights/last.pt, 22.5MB
Optimizer stripped from /data/zj/YOLO8Face/runs/detect/train7/weights/best.pt, 22.5MB

Validating /data/zj/YOLO8Face/runs/detect/train7/weights/best.pt...
Ultralytics YOLOv8.2.103 🚀 Python-3.8.10 torch-1.14.0a0+410ce96 CUDA:6 (NVIDIA GeForce RTX 3090, 24268MiB)
Model summary (fused): 168 layers, 11,125,971 parameters, 0 gradients
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 101/101 [00:21<00:00,  4.69it/s]
                   all       3225      39675      0.869       0.63      0.718        0.4
Speed: 0.2ms preprocess, 0.9ms inference, 0.0ms loss, 0.5ms postprocess per image
Results saved to /data/zj/YOLO8Face/runs/detect/train7
```

### YOLOv8s with I800 + E300

```shell
root@a988b27f02e9:/data/zj/YOLO8Face# python3 yolo8face_train.py --model yolov8s.pt --data ./yolo8face/cfg/datasets/widerface.yaml --device 5 --epochs 300 --imgsz 800 --batch 8                               
args: Namespace(data='./yolo8face/cfg/datasets/widerface.yaml', device=[5], model='yolov8s.pt', source='./yolo8face/assets') - unknown: ['--epochs', '300', '--imgsz', '800', '--batch', '8']
Ultralytics YOLOv8.2.103 🚀 Python-3.8.10 torch-1.14.0a0+410ce96 CUDA:5 (NVIDIA GeForce RTX 3090, 24268MiB)
WARNING ⚠️ Upgrade to torch>=2.0.0 for deterministic training.
engine/trainer: task=detect, mode=train, model=yolov8s.pt, data=./yolo8face/cfg/datasets/widerface.yaml, epochs=300, time=None, patience=100, batch=8, imgsz=800, save=True, save_period=-1, cache=False, devi$e=[5], workers=8, project=None, name=train9, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=Fa$se, amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=$00, half=False, dnn=False, plots=True, source=./yolo8face/assets, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=Fa$se, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dyna$ic=False, simplify=True, opset=None, workspace=4, nms=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, p$se=12.0, kobj=1.0, label_smoothing=0.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, $opy_paste=0.0, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=/data/zj/YOLO8Face/runs/detect/train9
TensorBoard: Start with 'tensorboard --logdir /data/zj/YOLO8Face/runs/detect/train9', view at http://localhost:6006/
Overriding model.yaml nc=80 with nc=1

                   from  n    params  module                                       arguments                       
  0                  -1  1       928  ultralytics.nn.modules.conv.Conv             [3, 32, 3, 2]                   
  1                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]                  
  2                  -1  1     29056  ultralytics.nn.modules.block.C2f             [64, 64, 1, True]               
  3                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]                 
  4                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]             
  5                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]             
  6                  -1  2    788480  ultralytics.nn.modules.block.C2f             [256, 256, 2, True]           
  7                  -1  1   1180672  ultralytics.nn.modules.conv.Conv             [256, 512, 3, 2]              
  8                  -1  1   1838080  ultralytics.nn.modules.block.C2f             [512, 512, 1, True]           
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]                 
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 12                  -1  1    591360  ultralytics.nn.modules.block.C2f             [768, 256, 1]                 
 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 15                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]                 
 16                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 18                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]                 
 19                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 21                  -1  1   1969152  ultralytics.nn.modules.block.C2f             [768, 512, 1]                 
 22        [15, 18, 21]  1   2116435  ultralytics.nn.modules.head.Detect           [1, [128, 256, 512]]          
Model summary: 225 layers, 11,135,987 parameters, 11,135,971 gradients

Transferred 349/355 items from pretrained weights
Freezing layer 'model.22.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks with YOLOv8n...
AMP: checks passed ✅
train: Scanning /data/zj/datasets/widerface/labels/train.cache... 12876 images, 0 backgrounds, 1 corrupt: 100%|██████████| 12876/12876 [00:00<?, ?it/s]train: WARNING ⚠️ /data/zj/datasets/widerface/images/train/2_Demonstration_Protesters_2_231.jpg: 1 duplicate labels removedtrain: WARNING ⚠️ /data/zj/datasets/widerface/images/train/37_Soccer_Soccer_37_851.jpg: 1 duplicate labels removedtrain: WARNING ⚠️ /data/zj/datasets/widerface/images/train/54_Rescue_rescuepeople_54_29.jpg: ignoring corrupt image/label: non-normalized or out of bounds coordinates [     1.0254]train: WARNING ⚠️ /data/zj/datasets/widerface/images/train/7_Cheering_Cheering_7_17.jpg: 1 duplicate labels removedval: Scanning /data/zj/datasets/widerface/labels/val.cache... 3226 images, 0 backgrounds, 1 corrupt: 100%|██████████| 3226/3226 [00:00<?, ?it/s]val: WARNING ⚠️ /data/zj/datasets/widerface/images/val/21_Festival_Festival_21_604.jpg: 1 duplicate labels removedval: WARNING ⚠️ /data/zj/datasets/widerface/images/val/39_Ice_Skating_iceskiing_39_583.jpg: ignoring corrupt image/label: non-normalized or out of bounds coordinates [      1.002]Plotting labels to /data/zj/YOLO8Face/runs/detect/train9/labels.jpg... 
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: SGD(lr=0.01, momentum=0.9) with parameter groups 57 weight(decay=0.0), 64 weight(decay=0.0005), 63 bias(decay=0.0)
TensorBoard: model graph visualization added ✅
Image sizes 800 train, 800 val
Using 8 dataloader workers
Logging results to /data/zj/YOLO8Face/runs/detect/train9
Starting training for 300 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/300      14.7G      1.718      1.196      1.188         32        800: 100%|██████████| 1610/1610 [02:41<00:00,  9.96it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 202/202 [00:22<00:00,  8.87it/s]
                   all       3225      39675      0.824       0.56      0.643       0.33
...
...
...
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    299/300      13.8G     0.9163     0.4072     0.8603         58        800: 100%|██████████| 1610/1610 [02:13<00:00, 12.09it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 202/202 [00:15<00:00, 13.44it/s]
                   all       3225      39675      0.881      0.677      0.767      0.431

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    300/300      18.4G     0.9163     0.4083       0.86         30        800: 100%|██████████| 1610/1610 [02:13<00:00, 12.02it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 202/202 [00:14<00:00, 13.48it/s]
                   all       3225      39675      0.881      0.678      0.767      0.431

300 epochs completed in 12.680 hours.
Optimizer stripped from /data/zj/YOLO8Face/runs/detect/train9/weights/last.pt, 22.5MB
Optimizer stripped from /data/zj/YOLO8Face/runs/detect/train9/weights/best.pt, 22.5MB

Validating /data/zj/YOLO8Face/runs/detect/train9/weights/best.pt...
Ultralytics YOLOv8.2.103 🚀 Python-3.8.10 torch-1.14.0a0+410ce96 CUDA:5 (NVIDIA GeForce RTX 3090, 24268MiB)
Model summary (fused): 168 layers, 11,125,971 parameters, 0 gradients
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 202/202 [00:24<00:00,  8.31it/s]
                   all       3225      39675      0.878       0.68      0.767      0.434
Speed: 0.2ms preprocess, 1.4ms inference, 0.0ms loss, 0.6ms postprocess per image
Results saved to /data/zj/YOLO8Face/runs/detect/train9
```

## YOLOv5

### YOLOv5n

```shell
root@a988b27f02e9:/data/zj/YOLO8Face# python3 yolo8face_train.py --model yolov5n.pt --data ./yolo8face/cfg/datasets/widerface.yaml --device 7
args: Namespace(data='./yolo8face/cfg/datasets/widerface.yaml', device=[7], model='yolov5n.pt', source='./yolo8face/assets') - unknown: []
Ultralytics YOLOv8.2.103 🚀 Python-3.8.10 torch-1.14.0a0+410ce96 CUDA:7 (NVIDIA GeForce RTX 3090, 24268MiB)
WARNING ⚠️ Upgrade to torch>=2.0.0 for deterministic training.
engine/trainer: task=detect, mode=train, model=yolov5n.pt, data=./yolo8face/cfg/datasets/widerface.yaml, epochs=100, time=None, patience=100, batch=16, imgsz=640, save=True, save_period=-1, cache=False, device=[7], workers=8, project=None, name=train, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=./yolo8face/assets, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=True, opset=None, workspace=4, nms=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, label_smoothing=0.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, copy_paste=0.0, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=/data/zj/YOLO8Face/runs/detect/train
TensorBoard: Start with 'tensorboard --logdir /data/zj/YOLO8Face/runs/detect/train', view at http://localhost:6006/
PRO TIP 💡 Replace 'model=yolov5n.pt' with new 'model=yolov5nu.pt'.
YOLOv5 'u' models are trained with https://github.com/ultralytics/ultralytics and feature improved performance vs standard YOLOv5 models trained with https://github.com/ultralytics/yolov5.

Overriding model.yaml nc=80 with nc=1

                   from  n    params  module                                       arguments                     
  0                  -1  1      1760  ultralytics.nn.modules.conv.Conv             [3, 16, 6, 2, 2]              
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]                
  2                  -1  1      4800  ultralytics.nn.modules.block.C3              [32, 32, 1]                   
  3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]                
  4                  -1  2     29184  ultralytics.nn.modules.block.C3              [64, 64, 2]                   
  5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  6                  -1  3    156928  ultralytics.nn.modules.block.C3              [128, 128, 3]                 
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              
  8                  -1  1    296448  ultralytics.nn.modules.block.C3              [256, 256, 1]                 
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]                 
 10                  -1  1     33024  ultralytics.nn.modules.conv.Conv             [256, 128, 1, 1]              
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1     90880  ultralytics.nn.modules.block.C3              [256, 128, 1, False]          
 14                  -1  1      8320  ultralytics.nn.modules.conv.Conv             [128, 64, 1, 1]               
 15                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 16             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 17                  -1  1     22912  ultralytics.nn.modules.block.C3              [128, 64, 1, False]           
 18                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
 19            [-1, 14]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 20                  -1  1     74496  ultralytics.nn.modules.block.C3              [128, 128, 1, False]          
 21                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
 22            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 23                  -1  1    296448  ultralytics.nn.modules.block.C3              [256, 256, 1, False]          
 24        [17, 20, 23]  1    751507  ultralytics.nn.modules.head.Detect           [1, [64, 128, 256]]           
YOLOv5n summary: 262 layers, 2,508,659 parameters, 2,508,643 gradients

Transferred 391/427 items from pretrained weights
Freezing layer 'model.24.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks with YOLOv8n...
AMP: checks passed ✅
train: Scanning /data/zj/datasets/widerface/labels/train.cache... 12876 images, 0 backgrounds, 1 corrupt: 100%|██████████| 12876/12876 [00:00<?, ?it/s]
train: WARNING ⚠️ /data/zj/datasets/widerface/images/train/2_Demonstration_Protesters_2_231.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /data/zj/datasets/widerface/images/train/37_Soccer_Soccer_37_851.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /data/zj/datasets/widerface/images/train/54_Rescue_rescuepeople_54_29.jpg: ignoring corrupt image/label: non-normalized or out of bounds coordinates [     1.0254]
train: WARNING ⚠️ /data/zj/datasets/widerface/images/train/7_Cheering_Cheering_7_17.jpg: 1 duplicate labels removed
val: Scanning /data/zj/datasets/widerface/labels/val.cache... 3226 images, 0 backgrounds, 1 corrupt: 100%|██████████| 3226/3226 [00:00<?, ?it/s]
val: WARNING ⚠️ /data/zj/datasets/widerface/images/val/21_Festival_Festival_21_604.jpg: 1 duplicate labels removed
val: WARNING ⚠️ /data/zj/datasets/widerface/images/val/39_Ice_Skating_iceskiing_39_583.jpg: ignoring corrupt image/label: non-normalized or out of bounds coordinates [      1.002]
Plotting labels to /data/zj/YOLO8Face/runs/detect/train/labels.jpg... 
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: SGD(lr=0.01, momentum=0.9) with parameter groups 69 weight(decay=0.0), 76 weight(decay=0.0005), 75 bias(decay=0.0)
TensorBoard: model graph visualization added ✅
Image sizes 640 train, 640 val
Using 8 dataloader workers
Logging results to /data/zj/YOLO8Face/runs/detect/train
Starting training for 100 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/100      15.4G      1.916       1.68      1.209        308        640: 100%|██████████| 805/805 [01:27<00:00,  9.18it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 101/101 [00:16<00:00,  6.03it/s]
                   all       3225      39675      0.726      0.438      0.489      0.239
...
...
...
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    100/100      20.4G      1.303       0.61     0.9397         63        640: 100%|██████████| 805/805 [01:11<00:00, 11.31it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 101/101 [00:10<00:00,  9.37it/s]
                   all       3225      39675      0.842      0.578       0.66      0.361

100 epochs completed in 2.319 hours.
Optimizer stripped from /data/zj/YOLO8Face/runs/detect/train/weights/last.pt, 5.3MB
Optimizer stripped from /data/zj/YOLO8Face/runs/detect/train/weights/best.pt, 5.3MB

Validating /data/zj/YOLO8Face/runs/detect/train/weights/best.pt...
Ultralytics YOLOv8.2.103 🚀 Python-3.8.10 torch-1.14.0a0+410ce96 CUDA:7 (NVIDIA GeForce RTX 3090, 24268MiB)
YOLOv5n summary (fused): 193 layers, 2,503,139 parameters, 0 gradients
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 101/101 [00:19<00:00,  5.25it/s]
                   all       3225      39675      0.842      0.577       0.66       0.36
Speed: 0.2ms preprocess, 0.5ms inference, 0.0ms loss, 0.5ms postprocess per image
Results saved to /data/zj/YOLO8Face/runs/detect/train
```

### YOLOv5s

```shell
root@a988b27f02e9:/data/zj/YOLO8Face# python3 yolo8face_train.py --model yolov5su.pt --data ./yolo8face/cfg/datasets/widerface.yaml --device 6
args: Namespace(data='./yolo8face/cfg/datasets/widerface.yaml', device=[6], model='yolov5su.pt', source='./yolo8face/assets') - unknown: []
Ultralytics YOLOv8.2.103 🚀 Python-3.8.10 torch-1.14.0a0+410ce96 CUDA:6 (NVIDIA GeForce RTX 3090, 24268MiB)
WARNING ⚠️ Upgrade to torch>=2.0.0 for deterministic training.
engine/trainer: task=detect, mode=train, model=yolov5su.pt, data=./yolo8face/cfg/datasets/widerface.yaml, epochs=100, time=None, patience=100, batch=16, imgsz=640, save=True, save_period=-1, cache=False, device=[6], workers=8, project=None, name=train2, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=./yolo8face/assets, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=True, opset=None, workspace=4, nms=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, label_smoothing=0.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, copy_paste=0.0, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=/data/zj/YOLO8Face/runs/detect/train2
TensorBoard: Start with 'tensorboard --logdir /data/zj/YOLO8Face/runs/detect/train2', view at http://localhost:6006/
Overriding model.yaml nc=80 with nc=1

                   from  n    params  module                                       arguments                     
  0                  -1  1      3520  ultralytics.nn.modules.conv.Conv             [3, 32, 6, 2, 2]              
  1                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]                
  2                  -1  1     18816  ultralytics.nn.modules.block.C3              [64, 64, 1]                   
  3                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  4                  -1  2    115712  ultralytics.nn.modules.block.C3              [128, 128, 2]                 
  5                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              
  6                  -1  3    625152  ultralytics.nn.modules.block.C3              [256, 256, 3]                 
  7                  -1  1   1180672  ultralytics.nn.modules.conv.Conv             [256, 512, 3, 2]              
  8                  -1  1   1182720  ultralytics.nn.modules.block.C3              [512, 512, 1]                 
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]                 
 10                  -1  1    131584  ultralytics.nn.modules.conv.Conv             [512, 256, 1, 1]              
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1    361984  ultralytics.nn.modules.block.C3              [512, 256, 1, False]          
 14                  -1  1     33024  ultralytics.nn.modules.conv.Conv             [256, 128, 1, 1]              
 15                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 16             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 17                  -1  1     90880  ultralytics.nn.modules.block.C3              [256, 128, 1, False]          
 18                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
 19            [-1, 14]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 20                  -1  1    296448  ultralytics.nn.modules.block.C3              [256, 256, 1, False]          
 21                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
 22            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 23                  -1  1   1182720  ultralytics.nn.modules.block.C3              [512, 512, 1, False]          
 24        [17, 20, 23]  1   2116435  ultralytics.nn.modules.head.Detect           [1, [128, 256, 512]]          
YOLOv5s summary: 262 layers, 9,122,579 parameters, 9,122,563 gradients

Transferred 421/427 items from pretrained weights
Freezing layer 'model.24.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks with YOLOv8n...
AMP: checks passed ✅
train: Scanning /data/zj/datasets/widerface/labels/train.cache... 12876 images, 0 backgrounds, 1 corrupt: 100%|██████████| 12876/12876 [00:00<?, ?it/s]
train: WARNING ⚠️ /data/zj/datasets/widerface/images/train/2_Demonstration_Protesters_2_231.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /data/zj/datasets/widerface/images/train/37_Soccer_Soccer_37_851.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /data/zj/datasets/widerface/images/train/54_Rescue_rescuepeople_54_29.jpg: ignoring corrupt image/label: non-normalized or out of bounds coordinates [     1.0254]
train: WARNING ⚠️ /data/zj/datasets/widerface/images/train/7_Cheering_Cheering_7_17.jpg: 1 duplicate labels removed
val: Scanning /data/zj/datasets/widerface/labels/val.cache... 3226 images, 0 backgrounds, 1 corrupt: 100%|██████████| 3226/3226 [00:00<?, ?it/s]
val: WARNING ⚠️ /data/zj/datasets/widerface/images/val/21_Festival_Festival_21_604.jpg: 1 duplicate labels removed
val: WARNING ⚠️ /data/zj/datasets/widerface/images/val/39_Ice_Skating_iceskiing_39_583.jpg: ignoring corrupt image/label: non-normalized or out of bounds coordinates [      1.002]
Plotting labels to /data/zj/YOLO8Face/runs/detect/train2/labels.jpg... 
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: SGD(lr=0.01, momentum=0.9) with parameter groups 69 weight(decay=0.0), 76 weight(decay=0.0005), 75 bias(decay=0.0)
TensorBoard: model graph visualization added ✅
Image sizes 640 train, 640 val
Using 8 dataloader workers
Logging results to /data/zj/YOLO8Face/runs/detect/train2
Starting training for 100 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/100      16.9G      1.809        1.2      1.175        308        640: 100%|██████████| 805/805 [01:39<00:00,  8.10it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 101/101 [00:16<00:00,  6.00it/s]
                   all       3225      39675      0.779      0.503      0.576      0.294
...
...
...
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     99/100      19.7G      1.159     0.5163     0.9036        204        640: 100%|██████████| 805/805 [01:21<00:00,  9.84it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 101/101 [00:11<00:00,  8.54it/s]
                   all       3225      39675      0.865      0.626      0.711      0.396

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    100/100      21.7G       1.16     0.5151     0.9022         63        640: 100%|██████████| 805/805 [01:19<00:00, 10.13it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 101/101 [00:11<00:00,  8.47it/s]
                   all       3225      39675      0.864      0.626      0.711      0.396

100 epochs completed in 2.649 hours.
Optimizer stripped from /data/zj/YOLO8Face/runs/detect/train2/weights/last.pt, 18.5MB
Optimizer stripped from /data/zj/YOLO8Face/runs/detect/train2/weights/best.pt, 18.5MB

Validating /data/zj/YOLO8Face/runs/detect/train2/weights/best.pt...
Ultralytics YOLOv8.2.103 🚀 Python-3.8.10 torch-1.14.0a0+410ce96 CUDA:6 (NVIDIA GeForce RTX 3090, 24268MiB)
YOLOv5s summary (fused): 193 layers, 9,111,923 parameters, 0 gradients
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 101/101 [00:20<00:00,  4.84it/s]
                   all       3225      39675      0.865      0.626      0.711      0.396
Speed: 0.1ms preprocess, 0.9ms inference, 0.0ms loss, 0.5ms postprocess per image
Results saved to /data/zj/YOLO8Face/runs/detect/train2
```