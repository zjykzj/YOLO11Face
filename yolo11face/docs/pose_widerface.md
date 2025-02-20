
# WIDERFACE

## yolov8n-pose

```shell
# python pose_widerface.py --model yolov8n-pose_widerface.pt --source ../datasets/widerface/images/val/ --folder_pict ../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt --save_txt true --conf 0.001 --iou 0.7 --max_det 300 --batch 1 --device 6
args: Namespace(data=None, device=[6], folder_pict='../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt', model='yolov8n-pose_widerface.pt', source='../datasets/widerface/images/val/') - unknown: ['--save_txt', 'true', '--conf', '0.001', '--iou', '0.7', '--max_det', '300', '--batch', '1']
{'model': 'yolov8n-pose_widerface.pt', 'data': None, 'device': [6], 'source': '../datasets/widerface/images/val/', 'folder_pict': '../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt', 'save_txt': True, 'conf': 0.001, 'iou': 0.7, 'max_det': 300, 'batch': 1, 'mode': 'predict'}
3226

Ultralytics YOLOv8.2.103 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:6 (NVIDIA GeForce RTX 3090, 24268MiB)
YOLOv8n-pose summary (fused): 187 layers, 3,078,128 parameters, 0 gradients, 8.3 GFLOPs
...
...
Speed: 2.1ms preprocess, 11.1ms inference, 1.4ms postprocess per image at shape (1, 3, 640, 448)
Results saved to /data/zj/YOLO8Face/runs/detect/predict2
0 label saved to /data/zj/YOLO8Face/runs/detect/predict2/labels
# cd widerface_evaluate/
# python3 evaluation.py -p ../runs/detect/predict2/labels/ -g ./ground_truth/
Reading Predictions : 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:00<00:00, 96.21it/s]
Processing easy: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:19<00:00,  3.18it/s]
Processing medium: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:19<00:00,  3.18it/s]
Processing hard: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:19<00:00,  3.16it/s]
==================== Results ====================
Easy   Val AP: 0.9407096374860786
Medium Val AP: 0.9203801823586393
Hard   Val AP: 0.7887785200794712
=================================================
```

## yolov8s-pose

```shell
# python pose_widerface.py --model yolov8s-pose_widerface.pt --source ../datasets/widerface/images/val/ --folder_pict ../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt --save_txt true --conf 0.001 --iou 0.7 --max_det 300 --batch 1 --device 0
args: Namespace(data=None, device=[0], folder_pict='../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt', model='yolov8s-pose_widerface.pt', source='../datasets/widerface/images/val/') - unknown: ['--save_txt', 'true', '--conf', '0.001', '--iou', '0.7', '--max_det', '300', '--batch', '1']
{'model': 'yolov8s-pose_widerface.pt', 'data': None, 'device': [0], 'source': '../datasets/widerface/images/val/', 'folder_pict': '../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt', 'save_txt': True, 'conf': 0.001, 'iou': 0.7, 'max_det': 300, 'batch': 1, 'mode': 'predict'}
3226

Ultralytics YOLOv8.2.103 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)
YOLOv8s-pose summary (fused): 187 layers, 11,413,344 parameters, 0 gradients, 29.4 GFLOPs
...
...
Speed: 2.1ms preprocess, 12.9ms inference, 1.6ms postprocess per image at shape (1, 3, 640, 448)
Results saved to /data/zj/YOLO8Face/runs/detect/predict
0 label saved to /data/zj/YOLO8Face/runs/detect/predict/labels
# cd widerface_evaluate/
# python3 evaluation.py -p ../runs/detect/predict/labels/ -g ./ground_truth/
Reading Predictions : 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:00<00:00, 116.17it/s]
Processing easy: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:18<00:00,  3.24it/s]
Processing medium: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:18<00:00,  3.24it/s]
Processing hard: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:19<00:00,  3.21it/s]
==================== Results ====================
Easy   Val AP: 0.9507304597200303
Medium Val AP: 0.9376731961249901
Hard   Val AP: 0.8283959392916
=================================================
```