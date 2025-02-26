
# YOLO8Face

## YOLOv8s-pose

```shell
# python3 pose_train.py --model yolov8s-pose.pt --data ./yolo11face/cfg/datasets/widerface-landmarks.yaml --epochs 300 --imgsz 800 --batch 8 --device 0
args: Namespace(data='./yolo11face/cfg/datasets/widerface-landmarks.yaml', device=[0], model='yolov8s-pose.pt', source=None) - unknown: ['--epochs', '300', '--imgsz', '800', '--batch', '8']
Ultralytics 8.3.75 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)
WARNING ⚠️ Upgrade to torch>=2.0.0 for deterministic training.
engine/trainer: task=pose, mode=train, model=yolov8s-pose.pt, data=./yolo11face/cfg/datasets/widerface-landmarks.yaml, epochs=300, time=None, patience=100, batch=8, imgsz=800, save=True, save_period=-1, cache=False, device=[0], workers=8, project=None, name=train3, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=True, opset=None, workspace=None, nms=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, copy_paste=0.0, copy_paste_mode=flip, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=/data/zj/YOLO11Face/runs/pose/train3
Overriding model.yaml kpt_shape=[17, 3] with kpt_shape=[5, 3]

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
 22        [15, 18, 21]  1   2404000  ultralytics.nn.modules.head.Pose             [1, [5, 3], [128, 256, 512]]  
YOLOv8s-pose summary: 250 layers, 11,423,552 parameters, 11,423,536 gradients, 29.6 GFLOPs

Transferred 361/397 items from pretrained weights
Freezing layer 'model.22.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Scanning /data/zj/datasets/widerface-landmarks/labels/train.cache... 10304 images, 0 backgrounds, 2 corrupt: 100%|██████████| 10304/10304 [00:00<?, ?it/s]
train: WARNING ⚠️ /data/zj/datasets/widerface-landmarks/images/train/37_Soccer_Soccer_37_851.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /data/zj/datasets/widerface-landmarks/images/train/54_Rescue_rescuepeople_54_29.jpg: ignoring corrupt image/label: negative label values [  -0.025391]
train: WARNING ⚠️ /data/zj/datasets/widerface-landmarks/images/train/7_Cheering_Cheering_7_17.jpg: ignoring corrupt image/label: negative label values [  -0.005371   -0.010742]
val: Scanning /data/zj/datasets/widerface-landmarks/labels/val.cache... 2576 images, 0 backgrounds, 0 corrupt: 100%|██████████| 2576/2576 [00:00<?, ?it/s]
Plotting labels to /data/zj/YOLO11Face/runs/pose/train3/labels.jpg... 
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: SGD(lr=0.01, momentum=0.9) with parameter groups 63 weight(decay=0.0), 73 weight(decay=0.0005), 72 bias(decay=0.0)
Image sizes 800 train, 800 val
Using 8 dataloader workers
Logging results to /data/zj/YOLO11Face/runs/pose/train3
Starting training for 300 epochs...

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      1/300      16.6G       1.72      1.287     0.5697      1.132      1.235        112        800: 100%|██████████| 1288/1288 [02:47<00:00,  7.71it/s][奸笑]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 161/161 [00:19<00:00,  8.35it/s]
                   all       2576      29560      0.832      0.548      0.622      0.329      0.753      0.397      0.411       0.37
...
...
      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
    299/300        15G     0.9582     0.1281    0.06786     0.4279     0.8741         54        800: 100%|██████████| 1288/1288 [02:23<00:00,  8.95it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 161/161 [00:17<00:00,  9.30it/s]
                   all       2576      29560      0.868      0.641      0.725      0.416       0.79      0.407      0.449      0.435

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
    300/300      14.7G     0.9552     0.1271    0.06373     0.4266     0.8719         36        800: 100%|██████████| 1288/1288 [02:25<00:00,  8.85it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 161/161 [00:17<00:00,  9.22it/s]
                   all       2576      29560      0.867      0.642      0.725      0.416       0.79      0.407      0.449      0.435

300 epochs completed in 13.717 hours.
Optimizer stripped from /data/zj/YOLO11Face/runs/pose/train3/weights/last.pt, 23.2MB
Optimizer stripped from /data/zj/YOLO11Face/runs/pose/train3/weights/best.pt, 23.2MB

Validating /data/zj/YOLO11Face/runs/pose/train3/weights/best.pt...
Ultralytics 8.3.75 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:7 (NVIDIA GeForce RTX 3090, 24268MiB)
YOLOv8s-pose summary (fused): 187 layers, 11,413,344 parameters, 0 gradients, 29.4 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 161/161 [00:32<00:00,  4.98it/s]
                   all       2576      29560      0.867      0.642      0.726      0.418       0.79      0.406      0.448      0.435
Speed: 0.2ms preprocess, 2.2ms inference, 0.0ms loss, 0.9ms postprocess per image
Results saved to /data/zj/YOLO11Face/runs/pose/train3
```

### YOLOv8n-pose

```shell
# python3 pose_train.py --model yolov8n-pose.pt --data ./yolo11face/cfg/datasets/widerface-landmarks.yaml --epochs 300 --imgsz 800 --batch 8 --device 0
args: Namespace(data='./yolo11face/cfg/datasets/widerface-landmarks.yaml', device=[0], model='yolov8n-pose.pt', source=None) - unknown: ['--epochs', '300', '--imgsz', '800', '--batch', '8']
Ultralytics 8.3.75 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)
WARNING ⚠️ Upgrade to torch>=2.0.0 for deterministic training.
engine/trainer: task=pose, mode=train, model=yolov8n-pose.pt, data=./yolo11face/cfg/datasets/widerface-landmarks.yaml, epochs=300, time=None, patience=100, batch=8, imgsz=800, save=True, save_period=-1, cache=False, device=[0], workers=8, project=None, name=train, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=True, opset=None, workspace=None, nms=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, copy_paste=0.0, copy_paste_mode=flip, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=/data/zj/YOLO11Face/runs/pose/train
Overriding model.yaml kpt_shape=[17, 3] with kpt_shape=[5, 3]

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
 22        [15, 18, 21]  1    823888  ultralytics.nn.modules.head.Pose             [1, [5, 3], [64, 128, 256]]   
YOLOv8n-pose summary: 250 layers, 3,083,424 parameters, 3,083,408 gradients, 8.4 GFLOPs

Transferred 361/397 items from pretrained weights
Freezing layer 'model.22.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Scanning /data/zj/datasets/widerface-landmarks/labels/train.cache... 10304 images, 0 backgrounds, 2 corrupt: 100%|██████████| 10304/10304 [00:00<?, ?it/s]
train: WARNING ⚠️ /data/zj/datasets/widerface-landmarks/images/train/37_Soccer_Soccer_37_851.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /data/zj/datasets/widerface-landmarks/images/train/54_Rescue_rescuepeople_54_29.jpg: ignoring corrupt image/label: negative label values [  -0.025391]
train: WARNING ⚠️ /data/zj/datasets/widerface-landmarks/images/train/7_Cheering_Cheering_7_17.jpg: ignoring corrupt image/label: negative label values [  -0.005371   -0.010742]
val: Scanning /data/zj/datasets/widerface-landmarks/labels/val.cache... 2576 images, 0 backgrounds, 0 corrupt: 100%|██████████| 2576/2576 [00:00<?, ?it/s]
Plotting labels to /data/zj/YOLO11Face/runs/pose/train/labels.jpg... 
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: SGD(lr=0.01, momentum=0.9) with parameter groups 63 weight(decay=0.0), 73 weight(decay=0.0005), 72 bias(decay=0.0)
Image sizes 800 train, 800 val
Using 8 dataloader workers
Logging results to /data/zj/YOLO11Face/runs/pose/train
Starting training for 300 epochs...

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      1/300      15.5G      1.827      1.588     0.6198      1.228      1.267        112        800: 100%|██████████| 1288/1288 [02:44<00:00,  7.85it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 161/161 [00:17<00:00,  9.27it/s]
                   all       2576      29560      0.799      0.485      0.551      0.273      0.764      0.379        0.4      0.344
...
...
```

