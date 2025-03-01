
# YOLOv8

## yolov8s

```shell
# python detect_widerface.py --model yolov8s_widerface.pt --source ../datasets/widerface/images/val/ --folder_pict ../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt --save_txt true --conf 0.001 --iou 0.6 --max_det 1000 --batch 1 --device 0
args: Namespace(data=None, device=[0], folder_pict='../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt', model='yolov8s_widerface.pt', source='../datasets/widerface/images/val/') - unknown: ['--save_txt', 'true', '--conf', '0.001', '--iou', '0.6', '--max_det', '1000', '--batch', '1']
{'model': 'yolov8s_widerface.pt', 'data': None, 'device': [0], 'source': '../datasets/widerface/images/val/', 'folder_pict': '../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt', 'save_txt': True, 'conf': 0.001, 'iou': 0.6, 'max_det': 1000, 'batch': 1, 'mode': 'predict'}
3226

Ultralytics 8.3.75 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)
Model summary (fused): 168 layers, 11,125,971 parameters, 0 gradients, 28.4 GFLOPs
...
...
Speed: 2.1ms preprocess, 12.3ms inference, 1.5ms postprocess per image at shape (1, 3, 640, 448)
Results saved to /data/zj/YOLO11Face/runs/detect/predict7
0 label saved to /data/zj/YOLO11Face/runs/detect/predict7/labels
# cd widerface_evaluate/
# python3 evaluation.py -p ../runs/detect/predict7/labels/ -g ./ground_truth/
Reading Predictions : 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:00<00:00, 109.04it/s]
Processing easy: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:19<00:00,  3.16it/s]
Processing medium: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:19<00:00,  3.20it/s]
Processing hard: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:19<00:00,  3.20it/s]
==================== Results ====================
Easy   Val AP: 0.9577126067050414
Medium Val AP: 0.9417563365965542
Hard   Val AP: 0.8453509960807319
=================================================
```