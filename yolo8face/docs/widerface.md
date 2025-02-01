
# WIDERFACE

## YOLOv8

```shell
# python widerface_detect.py --model yolov8n_widerface.pt --source ../datasets/widerface/images/val/ --folder_pict ../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt --save_txt true --conf 0.001 --iou 0.7 --max_det 300 --batch 1 --device 0
args: Namespace(data=None, device=[0], folder_pict='../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt', model='yolov8n_widerface.pt', source='../datasets/widerface/images/val/') - unknown: ['--save_txt', 'true', '--conf', '0.001', '--iou', '0.7', '--max_det', '300', '--batch', '1']
{'model': 'yolov8n_widerface.pt', 'data': None, 'device': [0], 'source': '../datasets/widerface/images/val/', 'folder_pict':'../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt', 'save_txt': True, 'conf': 0.001, 'iou': 0.7, 'max_det': 300, 'batch': 1, 'mode': 'predict'}
3226

Ultralytics YOLOv8.2.103 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)
Model summary (fused): 168 layers, 3,005,843 parameters, 0 gradients, 8.1 GFLOPs
...
...
Speed: 2.1ms preprocess, 9.8ms inference, 1.1ms postprocess per image at shape (1, 3, 640, 448)
Results saved to /data/zj/YOLO8Face/runs/detect/predict3
0 label saved to /data/zj/YOLO8Face/runs/detect/predict3/labels
# cd widerface_evaluate/
# python3 evaluation.py -p ../runs/detect/predict3/labels/ -g ./ground_truth/
Reading Predictions : 100%|██████████████████████████████████████████████████████████████████| 61/61 [00:00<00:00, 96.00it/s]
Processing easy: 100%|███████████████████████████████████████████████████████████████████████| 61/61 [00:19<00:00,  3.19it/s]
Processing medium: 100%|█████████████████████████████████████████████████████████████████████| 61/61 [00:18<00:00,  3.24it/s]
Processing hard: 100%|███████████████████████████████████████████████████████████████████████| 61/61 [00:18<00:00,  3.24it/s]
==================== Results ====================
Easy   Val AP: 0.9396273205641539
Medium Val AP: 0.9204178762213794
Hard   Val AP: 0.7899744594529312
=================================================
```

### YOLOv8s

* E100 + I640(TRAIN) + I640(EVAL)

```shell
# python widerface_detect.py --model yolov8s_widerface.pt --source ../datasets/widerface/images/val/ --folder_pict ../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt --save_txt true --conf 0.001 --iou 0.7 --max_det 300 --batch 1 --device 0
args: Namespace(data=None, device=[0], folder_pict='../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt', model='yolov8s_widerface.pt', source='../datasets/widerface/images/val/') - unknown: ['--save_txt', 'true', '--conf', '0.001', '--iou', '0.7', '--max_det', '300', '--batch', '1']
{'model': 'yolov8s_widerface.pt', 'data': None, 'device': [0], 'source': '../datasets/widerface/images/val/', 'folder_pict':'../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt', 'save_txt': True, 'conf': 0.001, 'iou': 0.7, 'max_det': 300, 'batch': 1, 'mode': 'predict'}
3226

Ultralytics YOLOv8.2.103 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)
Model summary (fused): 168 layers, 11,125,971 parameters, 0 gradients, 28.4 GFLOPs
...
...
Speed: 2.1ms preprocess, 10.7ms inference, 1.1ms postprocess per image at shape (1, 3, 640, 448)
Results saved to /data/zj/YOLO8Face/runs/detect/predict4
0 label saved to /data/zj/YOLO8Face/runs/detect/predict4/labels
# cd widerface_evaluate/
# python3 evaluation.py -p ../runs/detect/predict4/labels/ -g ./ground_truth/
Reading Predictions : 100%|█████████████████████████████████████████████████████████████████| 61/61 [00:00<00:00, 122.85it/s]
Processing easy: 100%|███████████████████████████████████████████████████████████████████████| 61/61 [00:18<00:00,  3.21it/s]
Processing medium: 100%|█████████████████████████████████████████████████████████████████████| 61/61 [00:18<00:00,  3.21it/s]
Processing hard: 100%|███████████████████████████████████████████████████████████████████████| 61/61 [00:18<00:00,  3.21it/s]
==================== Results ====================
Easy   Val AP: 0.9518557457057146
Medium Val AP: 0.9392209094364963
Hard   Val AP: 0.8298551429638127
=================================================
```

* E300 + I800(TRAIN) + I640(EVAL)

```shell
# python widerface_detect.py --model yolov8s_widerface.pt --source ../datasets/widerface/images/val/ --folder_pict ../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt --save_txt true --conf 0.001 --iou 0.7 --max_det 300 --batch 1 --device 0
args: Namespace(data=None, device=[0], folder_pict='../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt', model='yolov8s_widerface.pt', source='../datasets/widerface/images/val/') - unknown: ['--save_txt', 'true', '--conf', '0.001', '--iou', '0.7', '--max_det', '300', '--batch', '1']
{'model': 'yolov8s_widerface.pt', 'data': None, 'device': [0], 'source': '../datasets/widerface/images/val/', 'folder_pict':'../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt', 'save_txt': True, 'conf': 0.001, 'iou': 0.7, 'max_det': 300, 'batch': 1, 'mode': 'predict'}
3226

Ultralytics YOLOv8.2.103 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)
Model summary (fused): 168 layers, 11,125,971 parameters, 0 gradients, 28.4 GFLOPs
...
...
Speed: 2.3ms preprocess, 11.5ms inference, 1.3ms postprocess per image at shape (1, 3, 640, 448)
Results saved to /data/zj/YOLO8Face/runs/detect/predict
0 label saved to /data/zj/YOLO8Face/runs/detect/predict/labels
# cd widerface_evaluate/
# python3 evaluation.py -p ../runs/detect/predict/labels/ -g ./ground_truth/Reading Predictions : 100%|█████████████████████████████████████████████████████████████████| 61/61 [00:00<00:00, 129.47it/s]
Processing easy: 100%|███████████████████████████████████████████████████████████████████████| 61/61 [00:19<00:00,  3.12it/s]
Processing medium: 100%|█████████████████████████████████████████████████████████████████████| 61/61 [00:19<00:00,  3.21it/s]
Processing hard: 100%|███████████████████████████████████████████████████████████████████████| 61/61 [00:18<00:00,  3.24it/s]
==================== Results ====================
Easy   Val AP: 0.9581224522661581
Medium Val AP: 0.9426259663875606
Hard   Val AP: 0.8274812437947878
=================================================
```

## YOLOv5

### YOLOv5n

E100 + I640(TRAIN) + I640(EVAL)

* iou == 0.7

```shell
# python widerface_detect.py --model yolov5nu_widerface.pt --source ../datasets/widerface/images/val/ --folder_pict ../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt --save_txt true --conf 0.001 --iou 0.7 --max_det 300 --batch 1 --device 0
args: Namespace(data=None, device=[0], folder_pict='../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt', model='yolov5nu_widerface.pt', source='../datasets/widerface/images/val/') - unknown: ['--save_txt', 'true', '--conf', '0.001', '--iou', '0.7', '--max_det', '300', '--batch', '1']
{'model': 'yolov5nu_widerface.pt', 'data': None, 'device': [0], 'source': '../datasets/widerface/images/val/', 'folder_pict': '../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt', 'save_txt': True, 'conf': 0.001, 'iou': 0.7, 'max_det':300, 'batch': 1, 'mode': 'predict'}
3226

Ultralytics YOLOv8.2.103 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)
YOLOv5n summary (fused): 193 layers, 2,503,139 parameters, 0 gradients, 7.1 GFLOPs
...
...
Speed: 2.5ms preprocess, 12.4ms inference, 1.4ms postprocess per image at shape (1, 3, 640, 448)
Results saved to /data/zj/YOLO8Face/runs/detect/predict
0 label saved to /data/zj/YOLO8Face/runs/detect/predict/labels
# cd widerface_evaluate/
# python3 evaluation.py -p ../runs/detect/predict/labels/ -g ./ground_truth/
Reading Predictions : 100%|██████████████████████████████████████████████████████████████████| 61/61 [00:00<00:00, 87.02it/s]
Processing easy: 100%|███████████████████████████████████████████████████████████████████████| 61/61 [00:19<00:00,  3.18it/s]
Processing medium: 100%|█████████████████████████████████████████████████████████████████████| 61/61 [00:19<00:00,  3.18it/s]
Processing hard: 100%|███████████████████████████████████████████████████████████████████████| 61/61 [00:19<00:00,  3.20it/s]
==================== Results ====================
Easy   Val AP: 0.9353804234660791
Medium Val AP: 0.9148055309694593
Hard   Val AP: 0.7834106234606437
=================================================
```

* iou == 0.6

```shell
# python widerface_detect.py --model yolov5nu_widerface.pt --source ../datasets/widerface/images/val/ --folder_pict ../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt --save_txt true --conf 0.001 --iou 0.6 --max_det 300 --batch 1 --device 0
args: Namespace(data=None, device=[0], folder_pict='../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt', model='yolov5nu_widerface.pt', source='../datasets/widerface/images/val/') - unknown: ['--save_txt', 'true', '--conf', '0.001', '--iou', '0.6', '--max_det', '300', '--batch', '1']
{'model': 'yolov5nu_widerface.pt', 'data': None, 'device': [0], 'source': '../datasets/widerface/images/val/', 'folder_pict': '../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt', 'save_txt': True, 'conf': 0.001, 'iou': 0.6, 'max_det':300, 'batch': 1, 'mode': 'predict'}
3226

Ultralytics YOLOv8.2.103 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)
YOLOv5n summary (fused): 193 layers, 2,503,139 parameters, 0 gradients, 7.1 GFLOPs
...
...
Speed: 2.5ms preprocess, 12.4ms inference, 1.4ms postprocess per image at shape (1, 3, 640, 448)
Results saved to /data/zj/YOLO8Face/runs/detect/predict3
0 label saved to /data/zj/YOLO8Face/runs/detect/predict3/labels
# cd widerface_evaluate/
# python3 evaluation.py -p ../runs/detect/predict3/labels/ -g ./ground_truth/
Reading Predictions : 100%|██████████████████████████████████████████████████████████████████| 61/61 [00:00<00:00, 99.61it/s]
Processing easy: 100%|███████████████████████████████████████████████████████████████████████| 61/61 [00:18<00:00,  3.23it/s]
Processing medium: 100%|█████████████████████████████████████████████████████████████████████| 61/61 [00:18<00:00,  3.23it/s]
Processing hard: 100%|███████████████████████████████████████████████████████████████████████| 61/61 [00:18<00:00,  3.23it/s]
==================== Results ====================
Easy   Val AP: 0.9351074086237013
Medium Val AP: 0.9145630148235959
Hard   Val AP: 0.7868080988323036
=================================================
```

E300 + I800(TRAIN) + I640(EVAL)

```shell
# python widerface_detect.py --model yolov5nu_widerface.pt --source ../datasets/widerface/images/val/ --folder_pict ../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt --save_txt true --conf 0.001 --iou 0.6 --max_det 300 --batch 1 --device 0
args: Namespace(data=None, device=[0], folder_pict='../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt', model='yolov5nu_widerface.pt', source='../datasets/widerface/images/val/') - unknown: ['--save_txt', 'true', '--conf', '0.001', '--iou', '0.6', '--max_det', '300', '--batch', '1']
{'model': 'yolov5nu_widerface.pt', 'data': None, 'device': [0], 'source': '../datasets/widerface/images/val/', 'folder_pict': '../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt', 'save_txt': True, 'conf': 0.001, 'iou': 0.6, 'max_det':300, 'batch': 1, 'mode': 'predict'}
3226

Ultralytics YOLOv8.2.103 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)
YOLOv5n summary (fused): 193 layers, 2,503,139 parameters, 0 gradients, 7.1 GFLOPs
...
...
Speed: 2.1ms preprocess, 10.9ms inference, 1.1ms postprocess per image at shape (1, 3, 640, 448)
Results saved to /data/zj/YOLO8Face/runs/detect/predict3
0 label saved to /data/zj/YOLO8Face/runs/detect/predict3/labels
# cd widerface_evaluate/
# python3 evaluation.py -p ../runs/detect/predict3/labels/ -g ./ground_truth/
Reading Predictions : 100%|█████████████████████████████████████████████████████████████████| 61/61 [00:00<00:00, 117.11it/s]
Processing easy: 100%|███████████████████████████████████████████████████████████████████████| 61/61 [00:18<00:00,  3.25it/s]
Processing medium: 100%|█████████████████████████████████████████████████████████████████████| 61/61 [00:18<00:00,  3.25it/s]
Processing hard: 100%|███████████████████████████████████████████████████████████████████████| 61/61 [00:18<00:00,  3.25it/s]
==================== Results ====================
Easy   Val AP: 0.9395687977113857
Medium Val AP: 0.9182365826005703
Hard   Val AP: 0.7889410651237242
=================================================
```

### YOLOv5s

* E100 + I640(TRAIN) + I640(EVAL)

```shell
# python widerface_detect.py --model yolov5su_widerface.pt --source ../datasets/widerface/images/val/ --folder_pict ../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt --save_txt true --conf 0.001 --iou 0.7 --max_det 300 --batch 1 --device 0
args: Namespace(data=None, device=[0], folder_pict='../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt', model='yolov5su_widerface.pt', source='../datasets/widerface/images/val/') - unknown: ['--save_txt', 'true', '--conf', '0.001', '--iou', '0.7', '--max_det', '300', '--batch', '1']
{'model': 'yolov5su_widerface.pt', 'data': None, 'device': [0], 'source': '../datasets/widerface/images/val/', 'folder_pict': '../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt', 'save_txt': True, 'conf': 0.001, 'iou': 0.7, 'max_det':300, 'batch': 1, 'mode': 'predict'}
3226

Ultralytics YOLOv8.2.103 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)
YOLOv5s summary (fused): 193 layers, 9,111,923 parameters, 0 gradients, 23.8 GFLOPs
...
...
Speed: 2.1ms preprocess, 11.3ms inference, 1.1ms postprocess per image at shape (1, 3, 640, 448)
Results saved to /data/zj/YOLO8Face/runs/detect/predict2
0 label saved to /data/zj/YOLO8Face/runs/detect/predict2/labels
# cd widerface_evaluate/
# python3 evaluation.py -p ../runs/detect/predict2/labels/ -g ./ground_truth/
Reading Predictions : 100%|█████████████████████████████████████████████████████████████████| 61/61 [00:00<00:00, 115.72it/s]
Processing easy: 100%|███████████████████████████████████████████████████████████████████████| 61/61 [00:19<00:00,  3.09it/s]
Processing medium: 100%|█████████████████████████████████████████████████████████████████████| 61/61 [00:19<00:00,  3.09it/s]
Processing hard: 100%|███████████████████████████████████████████████████████████████████████| 61/61 [00:19<00:00,  3.08it/s]
==================== Results ====================
Easy   Val AP: 0.9529122847222152
Medium Val AP: 0.9388710665803213
Hard   Val AP: 0.8282292094234512
=================================================
```

* E300 + I800(TRAIN) + I640(EVAL)

```shell
# python widerface_detect.py --model yolov5su_widerface.pt --source ../datasets/widerface/images/val/ --folder_pict ../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt --save_txt true --conf 0.001 --iou 0.7 --max_det 300 --batch 1 --device 0
args: Namespace(data=None, device=[0], folder_pict='../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt', model='yolov5su_widerface.pt', source='../datasets/widerface/images/val/') - unknown: ['--save_txt', 'true', '--conf', '0.001', '--iou', '0.7', '--max_det', '300', '--batch', '1']
{'model': 'yolov5su_widerface.pt', 'data': None, 'device': [0], 'source': '../datasets/widerface/images/val/', 'folder_pict': '../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt', 'save_txt': True, 'conf': 0.001, 'iou': 0.7, 'max_det':300, 'batch': 1, 'mode': 'predict'}
3226

Ultralytics YOLOv8.2.103 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)
YOLOv5s summary (fused): 193 layers, 9,111,923 parameters, 0 gradients, 23.8 GFLOPs
...
...
Speed: 2.0ms preprocess, 11.4ms inference, 1.1ms postprocess per image at shape (1, 3, 640, 448)
Results saved to /data/zj/YOLO8Face/runs/detect/predict4
0 label saved to /data/zj/YOLO8Face/runs/detect/predict4/labels
# cd widerface_evaluate/
# python3 evaluation.py -p ../runs/detect/predict4/labels/ -g ./ground_truth/
Reading Predictions : 100%|█████████████████████████████████████████████████████████████████| 61/61 [00:00<00:00, 136.42it/s]
Processing easy: 100%|███████████████████████████████████████████████████████████████████████| 61/61 [00:18<00:00,  3.33it/s]
Processing medium: 100%|█████████████████████████████████████████████████████████████████████| 61/61 [00:18<00:00,  3.34it/s]
Processing hard: 100%|███████████████████████████████████████████████████████████████████████| 61/61 [00:18<00:00,  3.36it/s]
==================== Results ====================
Easy   Val AP: 0.9517961040437555
Medium Val AP: 0.9349518850038544
Hard   Val AP: 0.8247375288198857
=================================================
```