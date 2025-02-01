<!-- <div align="right">
  Language:
    🇺🇸
  <a title="Chinese" href="./README.zh-CN.md">🇨🇳</a>
</div> -->

<div align="center"><a title="" href="https://github.com/zjykzj/YOLO8Face"><img align="center" src="yolo8face/assets/logo/YOLO8Face.png" alt=""></a></div>

<p align="center">
  «YOLO8Face» combined YOLO5Face and YOLOv8 for face and keypoint detection
<br>
<br>
  <a href="https://github.com/RichardLitt/standard-readme"><img src="https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square" alt=""></a>
  <a href="https://conventionalcommits.org"><img src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg" alt=""></a>
  <a href="http://commitizen.github.io/cz-cli/"><img src="https://img.shields.io/badge/commitizen-friendly-brightgreen.svg" alt=""></a>
</p>

|                      |     ARCH     | GFLOPs |   Easy    |  Medium   |   Hard    |
|:--------------------:|:------------:|:------:|:---------:|:---------:|:---------:|
| **zjykzj/YOLO5Face** | yolov5s-v7.0 |  15.8  |   94.84   |   93.28   |   84.67   |
| **zjykzj/YOLO5Face** | yolov5n-v7.0 |  4.2   |   93.25   |   91.11   |   80.33   |
|                      |              |        |           |           |           |
| **zjykzj/YOLO8Face** |   yolov5su   |  23.8  |   95.29   |   93.89   |   82.82   |
| **zjykzj/YOLO8Face** |   yolov5nu   |  7.1   |   93.54   |   91.48   |   78.34   |
|                      |              |        |           |           |           |
| **zjykzj/YOLO8Face** |   yolov8s    |  28.4  |   95.81   |   94.26   |   82.75   |
| **zjykzj/YOLO8Face** |   yolov8n    |  8.1   |   93.96   |   92.04   |   79.00   |

*Using VGA resolution input images (the longer edge of the input image is scaled to 640, and the shorter edge is scaled accordingly)*

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Background](#background)
- [Installation](#installation)
- [Usage](#usage)
  - [Train](#train)
  - [Eval](#eval)
  - [Predict](#predict)
- [Maintainers](#maintainers)
- [Thanks](#thanks)
- [Contributing](#contributing)
- [License](#license)

## Background

According to the [YOLO5Face](https://github.com/zjykzj/YOLO5Face) implementation, it adds Landmarks-HEAD to YOLOv5 to achieve synchronous detection of faces and keypoints. The YOLOv8 is an upgraded version of YOLOv5, which naturally improves the performance of face and keypoint detection by combining YOLO5Face and YOLOv8.

Note: the latest implementation of YOLO8Face in our warehouse is entirely based on [ultralytics/ultralytics v8.2.103](https://github.com/ultralytics/ultralytics/releases/tag/v8.2.103)

## Installation

See [INSTALL.md](./yolo8face/docs/INSTALL.md)

## Usage  

Download the WIDERFACE dataset from http://shuoyang1213.me/WIDERFACE/, Then convert WIDERFACE dataset format.

```shell
$ python3 widerface2yolo.py ../datasets/widerface/WIDER_train/images ../datasets/widerface/wider_face_split/wider_face_train_bbx_gt.txt ../datasets/widerface/
$ python3 widerface2yolo.py ../datasets/widerface/WIDER_val/images ../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt ../datasets/widerface/
```

### Train

```shell
# YOLOv5s
$ python3 yolo8face_train.py --model yolov5su.pt --data ./yolo8face/cfg/datasets/widerface.yaml --epochs 300 --imgsz 800 --device 0
# YOLOv8s
$ python3 yolo8face_train.py --model yolov8s.pt --data ./yolo8face/cfg/datasets/widerface.yaml --epochs 300 --imgsz 800 --device 0
```

### Eval

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

### Predict

```shell
# python3 yolo8face_predict.py --model yolov8s_widerface.pt --source ./yolo8face/assets/widerface_val/ --imgsz 640 --device 0
args: Namespace(data=None, device=[0], model='yolov8s_widerface.pt', source='./yolo8face/assets/widerface_val/') - unknown: ['--imgsz', '640']

Ultralytics YOLOv8.2.103 🚀 Python-3.8.19 torch-1.12.1+cu113 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)
Model summary (fused): 168 layers, 11,125,971 parameters, 0 gradients, 28.4 GFLOPs
image 1/2 /data/zj/YOLO8Face/yolo8face/assets/widerface_val/39_Ice_Skating_iceskiing_39_351.jpg: 640x640 3 faces, 8.5ms
image 2/2 /data/zj/YOLO8Face/yolo8face/assets/widerface_val/9_Press_Conference_Press_Conference_9_632.jpg: 640x640 1 face, 8.5ms
Speed: 3.0ms preprocess, 8.5ms inference, 1.2ms postprocess per image at shape (2, 3, 640, 640)
Results saved to /data/zj/YOLO8Face/runs/detect/predict5
```

<p align="left"><img src="yolo8face/assets/predict/9_Press_Conference_Press_Conference_9_632.jpg" height="240"\>  <img src="yolo8face/assets/predict/39_Ice_Skating_iceskiing_39_351.jpg" height="240"\></p>

## Maintainers

* zhujian - *Initial work* - [zjykzj](https://github.com/zjykzj)

## Thanks

* [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
* [zjykzj/YOLO5Face](https://github.com/zjykzj/YOLO5Face)
* [deepcam-cn/yolov5-face](https://github.com/deepcam-cn/yolov5-face)

## Contributing

Anyone's participation is welcome! Open an [issue](https://github.com/zjykzj/YOLO8Face/issues) or submit PRs.

## License

[Apache License 2.0](LICENSE) © 2025 zjykzj