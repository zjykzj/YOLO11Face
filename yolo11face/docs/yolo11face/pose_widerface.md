
# YOLO11-pose

## yolo11n-pose

```shell
# python pose_widerface.py --model yolo11n-pose_widerface.pt --source ../datasets/widerface/images/val/ --folder_pict ../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt --save_txt true --imgsz 640 --conf 0.001 --iou 0.6 --max_det 1000 --batch 1 --device 7
args: Namespace(data=None, device=[7], folder_pict='../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt', model='yolo11n-pose_widerface.pt', source='../datasets/widerface/images/val/') - unknown: ['--save_txt', 'true', '--imgsz', '640', '--conf', '0.001', '--iou', '0.6', '--max_det', '1000', '--batch', '1']
{'model': 'yolo11n-pose_widerface.pt', 'data': None, 'device': [7], 'source': '../datasets/widerface/images/val/', 'folder_pict': '../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt', 'save_txt': True, 'imgsz': 640, 'conf': 0.001, 'iou': 0.6, 'max_det': 1000, 'batch': 1, 'mode': 'predict'}
3226

Ultralytics 8.3.75 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:7 (NVIDIA GeForce RTX 3090, 24268MiB)
YOLO11n-pose summary (fused): 257 layers, 2,654,632 parameters, 0 gradients, 6.6 GFLOPs
...
...
Speed: 2.1ms preprocess, 14.3ms inference, 1.4ms postprocess per image at shape (1, 3, 640, 448)
Results saved to /data/zj/YOLO11Face/runs/detect/predict2
0 label saved to /data/zj/YOLO11Face/runs/detect/predict2/labels
# cd widerface_evaluate/
# python3 evaluation.py -p ../runs/detect/predict2/labels/ -g ./ground_truth/
Reading Predictions : 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:00<00:00, 83.94it/s]
Processing easy: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:19<00:00,  3.13it/s]
Processing medium: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:19<00:00,  3.15it/s]
Processing hard: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:19<00:00,  3.14it/s]
==================== Results ====================
Easy   Val AP: 0.9461680018601685
Medium Val AP: 0.9255568821652825
Hard   Val AP: 0.8102161268183805
=================================================
```

## yolo11s-pose

```shell
# python pose_widerface.py --model yolo11s-pose_widerface.pt --source ../datasets/widerface/images/val/ --folder_pict ../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt --save_txt true --imgsz 640 --conf 0.001 --iou 0.6 --max_det 1000 --batch 1 --device 7
args: Namespace(data=None, device=[7], folder_pict='../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt', model='yolo11s-pose_widerface.pt', source='../datasets/widerface/images/val/') - unknown: ['--save_txt', 'true', '--imgsz', '640', '--conf', '0.001', '--iou', '0.6', '--max_det', '1000', '--batch', '1']
{'model': 'yolo11s-pose_widerface.pt', 'data': None, 'device': [7], 'source': '../datasets/widerface/images/val/', 'folder_pict': '../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt', 'save_txt': True, 'imgsz': 640, 'conf': 0.001, 'iou': 0.6, 'max_det': 1000, 'batch': 1, 'mode': 'predict'}
3226

Ultralytics 8.3.75 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:7 (NVIDIA GeForce RTX 3090, 24268MiB)
YOLO11s-pose summary (fused): 257 layers, 9,700,560 parameters, 0 gradients, 22.3 GFLOPs
...
...
Speed: 2.0ms preprocess, 14.4ms inference, 1.4ms postprocess per image at shape (1, 3, 640, 448)
Results saved to /data/zj/YOLO11Face/runs/detect/predict3
0 label saved to /data/zj/YOLO11Face/runs/detect/predict3/labels
# cd widerface_evaluate/
# python3 evaluation.py -p ../runs/detect/predict3/labels/ -g ./ground_truth/
Reading Predictions : 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:00<00:00, 115.26it/s]
Processing easy: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:19<00:00,  3.20it/s]
Processing medium: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:18<00:00,  3.22it/s]
Processing hard: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:18<00:00,  3.21it/s]
==================== Results ====================
Easy   Val AP: 0.9572097672239526
Medium Val AP: 0.9419027051471077
Hard   Val AP: 0.8523522955677869
=================================================
```