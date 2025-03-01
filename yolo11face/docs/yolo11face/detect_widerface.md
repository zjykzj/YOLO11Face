
# YOLO11

## yolo11s

```shell
# python detect_widerface.py --model yolo11s_widerface.pt --source ../datasets/widerface/images/val/ --folder_pict ../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt --save_txt true --conf 0.001 --iou 0.6 --max_det 1000 --batch 1 --device 0
args: Namespace(data=None, device=[0], folder_pict='../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt', model='yolo11s_widerface.pt', source='../datasets/widerface/images/val/') - unknown: ['--save_txt', 'true', '--conf', '0.001', '--iou', '0.6', '--max_det', '1000', '--batch', '1']
{'model': 'yolo11s_widerface.pt', 'data': None, 'device': [0], 'source': '../datasets/widerface/images/val/', 'folder_pict': '../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt', 'save_txt': True, 'conf': 0.001, 'iou': 0.6, 'max_det': 1000, 'batch': 1, 'mode': 'predict'}
3226

Ultralytics 8.3.75 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)
YOLO11s summary (fused): 238 layers, 9,413,187 parameters, 0 gradients, 21.3 GFLOPs
...
...
Speed: 2.1ms preprocess, 13.1ms inference, 1.2ms postprocess per image at shape (1, 3, 640, 448)
Results saved to /data/zj/YOLO11Face/runs/detect/predict6
0 label saved to /data/zj/YOLO11Face/runs/detect/predict6/labels
# cd widerface_evaluate/
# python3 evaluation.py -p ../runs/detect/predict6/labels/ -g ./ground_truth/
Reading Predictions : 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:00<00:00, 116.83it/s]
Processing easy: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:18<00:00,  3.24it/s]
Processing medium: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:18<00:00,  3.24it/s]
Processing hard: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:18<00:00,  3.23it/s]
==================== Results ====================
Easy   Val AP: 0.9555146054741577
Medium Val AP: 0.9391425630043154
Hard   Val AP: 0.8484696124027832
=================================================
```