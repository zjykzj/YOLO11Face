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

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Background](#background)
- [Installation](#installation)
- [Maintainers](#maintainers)
- [Thanks](#thanks)
- [Contributing](#contributing)
- [License](#license)

## Background

According to the [YOLO5Face](https://github.com/zjykzj/YOLO5Face) implementation, it adds Landmarks-HEAD to YOLOv5 to achieve synchronous detection of faces and keypoints. The YOLOv8 is an upgraded version of YOLOv5, which naturally improves the performance of face and keypoint detection by combining YOLO5Face and YOLOv8.

Note: the latest implementation of YOLO8Face in our warehouse is entirely based on [ultralytics/ultralytics v8.2.103](https://github.com/ultralytics/ultralytics/releases/tag/v8.2.103)

## Installation

```shell
docker run --gpus all -it --name yolo8face --shm-size=25g -v /data:/data nvcr.io/nvidia/pytorch:22.12-py3

pip3 install seaborn -i https://pypi.tuna.tsinghua.edu.cn/simple
```

<!-- ## Usage

### Data

Download the WIDERFACE dataset from http://shuoyang1213.me/WIDERFACE/, Then convert WIDERFACE dataset format.

```shell
python3 widerface2yolo.py ../datasets/widerface/WIDER_train/images ../datasets/widerface/wider_face_split/wider_face_train_bbx_gt.txt ../datasets/widerface/
python3 widerface2yolo.py ../datasets/widerface/WIDER_val/images ../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt ../datasets/widerface/
```

### Train

```shell
# YOLOv5s
$ python3 yolo8face_train.py --model yolov5su.pt --data ./yolo8face/cfg/datasets/widerface.yaml --device 0
# YOLOv8s
$ python3 yolo8face_train.py --model yolov8s.pt --data ./yolo8face/cfg/datasets/widerface.yaml --device 0
$ python3 yolo8face_train.py --model yolov8s.pt --data ./yolo8face/cfg/datasets/widerface.yaml --device 0 --epochs 300 --imgsz 800 --batch 8
```

### Eval

```shell
root@a988b27f02e9:/data/zj/YOLO8Face# python3 yolo8face_eval.py --model ./runs/detect/train/weights/best.pt --data ./yolo8face/cfg/datasets/widerface.yaml --device 0
args: Namespace(data='./yolo8face/cfg/datasets/widerface.yaml', device=[0], model='./runs/detect/train/weights/best.pt', source='./yolo8face/assets') - unknown: []
Ultralytics YOLOv8.2.103 🚀 Python-3.8.10 torch-1.14.0a0+410ce96 CUDA:0 (NVIDIA GeForce RTX 3090, 24268MiB)
YOLOv5n summary (fused): 193 layers, 2,503,139 parameters, 0 gradients
val: Scanning /data/zj/datasets/widerface/labels/val.cache... 3226 images, 0 backgrounds, 1 corrupt: 100%|██████████| 3226/3226 [00:00<?, ?it/s]
val: WARNING ⚠️ /data/zj/datasets/widerface/images/val/21_Festival_Festival_21_604.jpg: 1 duplicate labels removed
val: WARNING ⚠️ /data/zj/datasets/widerface/images/val/39_Ice_Skating_iceskiing_39_583.jpg: ignoring corrupt image/label: non-normalized or out of bounds coordinates [      1.002]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 202/202 [00:21<00:00,  9.27it/s]
                   all       3225      39675      0.841      0.579      0.661      0.362
Speed: 0.2ms preprocess, 1.0ms inference, 0.0ms loss, 0.6ms postprocess per image
Results saved to /data/zj/YOLO8Face/runs/detect/val


# python widerface_detect.py --weights ./runs/train/exp4-yolov5s-e250-img800.pt --source ../datasets/widerface/images/val/ --folder_pict ../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt --conf-thres 0.001 --iou-thres 0.6 --save-txt --save-conf --device 0
...
YOLOv5s summary: 157 layers, 7012822 parameters, 0 gradients, 15.8 GFLOPs
...
Speed: 0.3ms pre-process, 9.0ms inference, 0.9ms NMS per image at shape (1, 3, 640, 640)
Results saved to runs/detect/exp5
0 labels saved to runs/detect/exp5/labels
# cd widerface_evaluate/
# python3 evaluation.py -p ../runs/detect/exp5/labels/ -g ./ground_truth/
Reading Predictions : 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:00<00:00, 94.45it/s]
Processing easy: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:19<00:00,  3.13it/s]
Processing medium: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:19<00:00,  3.12it/s]
Processing hard: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:19<00:00,  3.13it/s]
==================== Results ====================
Easy   Val AP: 0.9520941964576021
Medium Val AP: 0.9341770033021547
Hard   Val AP: 0.8403303849682994
=================================================
```

### Predict

```shell
python detect.py --weights ./runs/exp4-yolov5s-e250-img800.pt --source assets/selfie.jpg --imgsz 2048 --conf-thres 0.25 --iou-thres 0.45 --max-det 3000 --hide-labels --hide-conf

python3 yolo8face_predict.py --model ./runs/yolov8s_widerface.pt --source ./yolo8face/assets/ --device 0
```

![](./assets/results/selfie.jpg) -->

## Maintainers

* zhujian - *Initial work* - [zjykzj](https://github.com/zjykzj)

## Thanks

* [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
* [zjykzj/YOLO5Face](https://github.com/zjykzj/YOLO5Face)
* [deepcam-cn/yolov5-face](https://github.com/deepcam-cn/yolov5-face)

## Contributing

Anyone's participation is welcome! Open an [issue](https://github.com/zjykzj/YOLO8Face/issues) or submit PRs.

## License

[Apache License 2.0](LICENSE) © 2024 zjykzj