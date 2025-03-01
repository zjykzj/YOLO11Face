
# YOLO11-pose

## YOLO11s-pose

```shell
# python3 pose_train.py --model yolo11s-pose.pt --data ./yolo11face/cfg/datasets/widerface-landmarks.yaml --epochs 300 --imgsz 800 --batch 8 --device 0
args: Namespace(data='./yolo11face/cfg/datasets/widerface-landmarks.yaml', device=[0], model='yolo11s-pose.pt', source=None) - unknown: ['--epochs', '300', '--imgsz', '800', '--batch', '8']
Ultralytics 8.3.75 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)
WARNING ⚠️ Upgrade to torch>=2.0.0 for deterministic training.
engine/trainer: task=pose, mode=train, model=yolo11s-pose.pt, data=./yolo11face/cfg/datasets/widerface-landmarks.yaml, epochs=300, time=None, patience=100, batch=8, imgsz=800, save=True, save_period=-1, cache=False, device=[0], workers=8, project=None, name=train4, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=True, opset=None, workspace=None, nms=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, copy_paste=0.0, copy_paste_mode=flip, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=/data/zj/YOLO11Face/runs/pose/train4
Overriding model.yaml kpt_shape=[17, 3] with kpt_shape=[5, 3]
Overriding model.yaml nc=80 with nc=1

                   from  n    params  module                                       arguments                     
  0                  -1  1       928  ultralytics.nn.modules.conv.Conv             [3, 32, 3, 2]                 
  1                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]                
  2                  -1  1     26080  ultralytics.nn.modules.block.C3k2            [64, 128, 1, False, 0.25]     
  3                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
  4                  -1  1    103360  ultralytics.nn.modules.block.C3k2            [128, 256, 1, False, 0.25]    
  5                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
  6                  -1  1    346112  ultralytics.nn.modules.block.C3k2            [256, 256, 1, True]           
  7                  -1  1   1180672  ultralytics.nn.modules.conv.Conv             [256, 512, 3, 2]              
  8                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]                 
 10                  -1  1    990976  ultralytics.nn.modules.block.C2PSA           [512, 512, 1]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1    443776  ultralytics.nn.modules.block.C3k2            [768, 256, 1, False]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  1    127680  ultralytics.nn.modules.block.C3k2            [512, 128, 1, False]          
 17                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  1    345472  ultralytics.nn.modules.block.C3k2            [384, 256, 1, False]          
 20                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1   1511424  ultralytics.nn.modules.block.C3k2            [768, 512, 1, True]           
 23        [16, 19, 22]  1   1107360  ultralytics.nn.modules.head.Pose             [1, [5, 3], [128, 256, 512]]  
YOLO11s-pose summary: 344 layers, 9,715,744 parameters, 9,715,728 gradients, 22.5 GFLOPs

Transferred 505/541 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Scanning /data/zj/datasets/widerface-landmarks/labels/train.cache... 10304 images, 0 backgrounds, 2 corrupt: 100%|██████████| 10304/10304 [00:00<?, ?it/s]
train: WARNING ⚠️ /data/zj/datasets/widerface-landmarks/images/train/37_Soccer_Soccer_37_851.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /data/zj/datasets/widerface-landmarks/images/train/54_Rescue_rescuepeople_54_29.jpg: ignoring corrupt image/label: negative label values [  -0.025391]
train: WARNING ⚠️ /data/zj/datasets/widerface-landmarks/images/train/7_Cheering_Cheering_7_17.jpg: ignoring corrupt image/label: negative label values [  -0.005371   -0.010742]
val: Scanning /data/zj/datasets/widerface-landmarks/labels/val.cache... 2576 images, 0 backgrounds, 0 corrupt: 100%|██████████| 2576/2576 [00:00<?, ?it/s]
Plotting labels to /data/zj/YOLO11Face/runs/pose/train4/labels.jpg... 
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: SGD(lr=0.01, momentum=0.9) with parameter groups 87 weight(decay=0.0), 97 weight(decay=0.0005), 96 bias(decay=0.0)
Image sizes 800 train, 800 val
Using 8 dataloader workers
Logging results to /data/zj/YOLO11Face/runs/pose/train4
Starting training for 300 epochs...

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      1/300      16.9G      1.704      1.227      0.565      1.113      1.181        112        800: 100%|██████████| 1288/1288 [03:17<00:00,  6.52it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 161/161 [00:19<00:00,  8.08it/s]
                   all       2576      29560      0.838      0.574      0.643      0.342      0.756      0.403      0.419       0.38
...
...
```

### YOLO11n-pose

```shell
# python3 pose_train.py --model yolo11n-pose.pt --data ./yolo11face/cfg/datasets/widerface-landmarks.yaml --epochs 300 --imgsz 800 --batch 8 --device 0
args: Namespace(data='./yolo11face/cfg/datasets/widerface-landmarks.yaml', device=[0], model='yolo11n-pose.pt', source=None) - unknown: ['--epochs', '300', '--imgsz', '800', '--batch', '8']
Ultralytics 8.3.75 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)
WARNING ⚠️ Upgrade to torch>=2.0.0 for deterministic training.
engine/trainer: task=pose, mode=train, model=yolo11n-pose.pt, data=./yolo11face/cfg/datasets/widerface-landmarks.yaml, epochs=300, time=None, patience=100, batch=8, imgsz=800, save=True, save_period=-1, cache=False, device=[0], workers=8, project=None, name=train2, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=True, opset=None, workspace=None, nms=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, copy_paste=0.0, copy_paste_mode=flip, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=/data/zj/YOLO11Face/runs/pose/train2
Overriding model.yaml kpt_shape=[17, 3] with kpt_shape=[5, 3]
Overriding model.yaml nc=80 with nc=1

                   from  n    params  module                                       arguments                     
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]                 
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]                
  2                  -1  1      6640  ultralytics.nn.modules.block.C3k2            [32, 64, 1, False, 0.25]      
  3                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
  4                  -1  1     26080  ultralytics.nn.modules.block.C3k2            [64, 128, 1, False, 0.25]     
  5                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
  6                  -1  1     87040  ultralytics.nn.modules.block.C3k2            [128, 128, 1, True]           
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              
  8                  -1  1    346112  ultralytics.nn.modules.block.C3k2            [256, 256, 1, True]           
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]                 
 10                  -1  1    249728  ultralytics.nn.modules.block.C2PSA           [256, 256, 1]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1    111296  ultralytics.nn.modules.block.C3k2            [384, 128, 1, False]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  1     32096  ultralytics.nn.modules.block.C3k2            [256, 64, 1, False]           
 17                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  1     86720  ultralytics.nn.modules.block.C3k2            [192, 128, 1, False]          
 20                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1    378880  ultralytics.nn.modules.block.C3k2            [384, 256, 1, True]           
 23        [16, 19, 22]  1    503248  ultralytics.nn.modules.head.Pose             [1, [5, 3], [64, 128, 256]]   
YOLO11n-pose summary: 344 layers, 2,662,416 parameters, 2,662,400 gradients, 6.7 GFLOPs

Transferred 505/541 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Scanning /data/zj/datasets/widerface-landmarks/labels/train.cache... 10304 images, 0 backgrounds, 2 corrupt: 100%|██████████| 10304/10304 [00:00<?, ?it/s]
train: WARNING ⚠️ /data/zj/datasets/widerface-landmarks/images/train/37_Soccer_Soccer_37_851.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /data/zj/datasets/widerface-landmarks/images/train/54_Rescue_rescuepeople_54_29.jpg: ignoring corrupt image/label: negative label values [  -0.025391]
train: WARNING ⚠️ /data/zj/datasets/widerface-landmarks/images/train/7_Cheering_Cheering_7_17.jpg: ignoring corrupt image/label: negative label values [  -0.005371   -0.010742]
val: Scanning /data/zj/datasets/widerface-landmarks/labels/val.cache... 2576 images, 0 backgrounds, 0 corrupt: 100%|██████████| 2576/2576 [00:00<?, ?it/s]
Plotting labels to /data/zj/YOLO11Face/runs/pose/train2/labels.jpg... 
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: SGD(lr=0.01, momentum=0.9) with parameter groups 87 weight(decay=0.0), 97 weight(decay=0.0005), 96 bias(decay=0.0)
Image sizes 800 train, 800 val
Using 8 dataloader workers
Logging results to /data/zj/YOLO11Face/runs/pose/train2
Starting training for 300 epochs...

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      1/300      15.7G       1.81      1.616     0.6296      1.228       1.25        112        800: 100%|██████████| 1288/1288 [03:05<00:00,  6.95it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 161/161 [00:18<00:00,  8.56it/s]
                   all       2576      29560      0.797      0.511       0.58      0.301      0.757      0.383      0.403      0.351
...
...
```

