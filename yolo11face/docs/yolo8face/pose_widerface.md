
# WIDERFACE 

## yolov8n-pose

### yolov8n-pose + IOU 0.7 + MaxDet 300

```shell
# python pose_widerface.py --model yolov8n-pose_widerface.pt --source ../datasets/widerface/images/val/ --folder_pict ../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt --save_txt true --conf 0.001 --iou 0.7 --max_det 300 --batch 1 --device 0
args: Namespace(data=None, device=[0], folder_pict='../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt', model='yolov8s-pose_widerface.pt', source='../datasets/widerface/images/val/') - unknown: ['--save_txt', 'true', '--conf', '0.001', '--iou', '0.7', '--max_det', '300', '--batch', '1']
...
...
Ultralytics 8.3.75 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)
YOLOv8n-pose summary (fused): 187 layers, 3,078,128 parameters, 0 gradients, 8.3 GFLOPs
...
...
Speed: 2.2ms preprocess, 11.1ms inference, 1.6ms postprocess per image at shape (1, 3, 640, 448)
Results saved to /data/zj/YOLO11Face/runs/detect/predict
0 label saved to /data/zj/YOLO11Face/runs/detect/predict/labels
# cd widerface_evaluate/
# python3 evaluation.py -p ../runs/detect/predict/labels/ -g ./ground_truth/
Reading Predictions : 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:00<00:00, 110.20it/s]
Processing easy: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:19<00:00,  3.20it/s]
Processing medium: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:19<00:00,  3.21it/s]
Processing hard: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:19<00:00,  3.19it/s]
==================== Results ====================
Easy   Val AP: 0.9467362929219146
Medium Val AP: 0.9250306949831006
Hard   Val AP: 0.7921294638844231
```

### yolov8n-pose + IOU 0.6 + MaxDet 300

```shell
# python pose_widerface.py --model yolov8n-pose_widerface.pt --source ../datasets/widerface/images/val/ --folder_pict ../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt --save_txt true --conf 0.001 --iou 0.6 --max_det 300 --batch 1 --device 0
args: Namespace(data=None, device=[0], folder_pict='../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt', model='yolov8s-pose_widerface.pt', source='../datasets/widerface/images/val/') - unknown: ['--save_txt', 'true', '--conf', '0.001', '--iou', '0.6', '--max_det', '300', '--batch', '1']
...
...
Ultralytics 8.3.75 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)
YOLOv8n-pose summary (fused): 187 layers, 3,078,128 parameters, 0 gradients, 8.3 GFLOPs
...
...
Speed: 2.1ms preprocess, 11.5ms inference, 1.6ms postprocess per image at shape (1, 3, 640, 448)
Results saved to /data/zj/YOLO11Face/runs/detect/predict2
0 label saved to /data/zj/YOLO11Face/runs/detect/predict2/labels
# cd widerface_evaluate/
# python3 evaluation.py -p ../runs/detect/predict2/labels/ -g ./ground_truth/
Reading Predictions : 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:00<00:00, 120.82it/s]
Processing easy: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:18<00:00,  3.27it/s]
Processing medium: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:18<00:00,  3.27it/s]
Processing hard: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:18<00:00,  3.26it/s]
==================== Results ====================
Easy   Val AP: 0.946943980762075
Medium Val AP: 0.9252478129654433
Hard   Val AP: 0.7952281714537119
=================================================
```

### yolov8n-pose + IOU 0.6 + MaxDet 1000

```shell
# python pose_widerface.py --model yolov8n-pose_widerface.pt --source ../datasets/widerface/images/val/ --folder_pict ../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt --save_txt true --conf 0.001 --iou 0.6 --max_det 1000 --batch 1 --device 0
args: Namespace(data=None, device=[0], folder_pict='../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt', model='yolov8n-pose_widerface.pt', source='../datasets/widerface/images/val/') - unknown: ['--save_txt', 'true', '--conf', '0.001', '--iou', '0.6', '--max_det', '1000', '--batch', '1']
{'model': 'yolov8n-pose_widerface.pt', 'data': None, 'device': [0], 'source': '../datasets/widerface/images/val/', 'folder_pict': '../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt', 'save_txt': True, 'conf': 0.001, 'iou': 0.6, 'max_det': 1000, 'batch': 1, 'mode': 'predict'}
3226

Ultralytics 8.3.75 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)
YOLOv8n-pose summary (fused): 187 layers, 3,078,128 parameters, 0 gradients, 8.3 GFLOPs
...
...
Speed: 2.0ms preprocess, 11.2ms inference, 1.4ms postprocess per image at shape (1, 3, 640, 448)
Results saved to /data/zj/YOLO11Face/runs/detect/predict3
0 label saved to /data/zj/YOLO11Face/runs/detect/predict3/labels
# cd widerface_evaluate/
# python3 evaluation.py -p ../runs/detect/predict3/labels/ -g ./ground_truth/
Reading Predictions : 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:00<00:00, 91.86it/s]
Processing easy: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:19<00:00,  3.08it/s]
Processing medium: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:19<00:00,  3.10it/s]
Processing hard: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:19<00:00,  3.09it/s]
==================== Results ====================
Easy   Val AP: 0.9461407631427757
Medium Val AP: 0.9245992916725901
Hard   Val AP: 0.80981983991975
=================================================
```

## yolov8s-pose

* yolov8s-pose + IOU 0.6 + MaxDet 1000

```shell
# python pose_widerface.py --model yolov8s-pose_widerface.pt --source ../datasets/widerface/images/val/ --folder_pict ../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt --save_txt true --imgsz 640 --conf 0.001 --iou 0.6 --max_det 1000 --batch 1 --device 0
args: Namespace(data=None, device=[0], folder_pict='../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt', model='yolov8s-pose_widerface.pt', source='../datasets/widerface/images/val/') - unknown: ['--save_txt', 'true', '--imgsz', '640', '--conf', '0.001', '--iou', '0.6', '--max_det', '1000', '--batch', '1']
{'model': 'yolov8s-pose_widerface.pt', 'data': None, 'device': [0], 'source': '../datasets/widerface/images/val/', 'folder_pict': '../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt', 'save_txt': True, 'imgsz': 640, 'conf': 0.001, 'iou': 0.6, 'max_det': 1000, 'batch': 1, 'mode': 'predict'}
3226

Ultralytics 8.3.75 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)
YOLOv8s-pose summary (fused): 187 layers, 11,413,344 parameters, 0 gradients, 29.4 GFLOPs
...
...
Speed: 2.0ms preprocess, 12.2ms inference, 1.4ms postprocess per image at shape (1, 3, 640, 448)
Results saved to /data/zj/YOLO11Face/runs/detect/predict
0 label saved to /data/zj/YOLO11Face/runs/detect/predict/labels
# cd widerface_evaluate/
# python3 evaluation.py -p ../runs/detect/predict/labels/ -g ./ground_truth/
Reading Predictions : 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:00<00:00, 116.63it/s]
Processing easy: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:19<00:00,  3.13it/s]
Processing medium: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:19<00:00,  3.16it/s]
Processing hard: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:19<00:00,  3.16it/s]
==================== Results ====================
Easy   Val AP: 0.955017081702656
Medium Val AP: 0.9394564958802335
Hard   Val AP: 0.846468041870603
=================================================
```