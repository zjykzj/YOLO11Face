# -*- coding: utf-8 -*-

"""
@Time    : 2025/2/02 18:08
@File    : widerface-landmarks2coco-pose.py
@Author  : zj
@Description:

Download the WIDERFACE dataset: http://shuoyang1213.me/WIDERFACE/

Download face and keypoint annotations: https://drive.google.com/file/d/1tU_IjyOwGQfGNUvZGwWWM4SwxKp2PUQ8/view?usp=sharing

Custom WIDERFACE training dataset, divided into keypoint detection training dataset and validation dataset in an 8:2 ratio.

Usage - Convert the WIDERFACE dataset format to YOLO-pose:
    $ python3 widerface2yolo-pose.py ../datasets/widerface/WIDER_train/images ../datasets/widerface/retinaface_gt_v1.1/train/label.txt ../datasets/widerface-landmarks/

"""

import os
import cv2
import random
import shutil
import argparse

import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="WiderFace2YOLO-pose")
    parser.add_argument('image', metavar='IMAGE', type=str, help='WiderFace image root.')
    parser.add_argument('label', metavar='LABEL', type=str, help='WiderFace label path.')

    parser.add_argument('dst', metavar='DST', type=str, help='YOLOLike data root.')

    args = parser.parse_args()
    print("args:", args)
    return args


def load_label(file_path):
    data = []
    current_image_data = None

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#'):
                # 新的图像路径开始
                if current_image_data is not None:
                    data.append(current_image_data)
                image_path = line.strip()[2:]
                current_image_data = {'image_path': image_path, 'annotations': []}
            else:
                parts = line.split(' ')
                assert len(parts) > 5
                bbox = list(map(int, parts[:4]))

                # 从第5个元素开始，直到倒数第二个元素，每2个元素形成一个关键点
                keypoints = [
                    (float(parts[i]), float(parts[i + 1]), float(parts[i + 2])) for i in range(4, len(parts) - 1, 3)
                ]
                assert len(keypoints) == 5, keypoints
                confidence = float(parts[-1])

                annotation = {
                    'bbox': bbox,
                    'keypoints': keypoints,
                    'confidence': confidence
                }
                current_image_data['annotations'].append(annotation)

        # 添加最后一个图像的信息
        if current_image_data is not None:
            data.append(current_image_data)

    return data


def create_dataset(results, img_root, dst_root, is_train=False):
    if is_train:
        dst_image_root = os.path.join(dst_root, "images", "train")
        dst_label_root = os.path.join(dst_root, "labels", "train")
    else:
        dst_image_root = os.path.join(dst_root, "images", "val")
        dst_label_root = os.path.join(dst_root, "labels", "val")
    if not os.path.exists(dst_image_root):
        os.makedirs(dst_image_root)
    if not os.path.exists(dst_label_root):
        os.makedirs(dst_label_root)

    cls_id = 0
    for result in tqdm(results):
        image_path = os.path.join(img_root, result["image_path"])
        assert os.path.isfile(image_path), image_path

        image = cv2.imread(image_path)
        height, width, channels = image.shape

        labels = []
        for anno in result['annotations']:
            label = []

            assert isinstance(anno, dict)
            x1, y1, box_w, box_h = anno['bbox']
            x_c = format(1.0 * (x1 + box_w / 2) / width, ".6f")
            y_c = format(1.0 * (y1 + box_h / 2) / height, ".6f")
            box_w = format(1.0 * box_w / width, ".6f")
            box_h = format(1.0 * box_h / height, ".6f")
            label.extend([int(cls_id), x_c, y_c, box_w, box_h])

            for point in anno['keypoints']:
                px, py, visible = point
                if px >= width or py >= height or px <= 0 or py <= 0:
                    px = 0.
                    py = 0.
                    visible = -1.0

                if visible == 0:
                    assert 0 <= px < width and 0 <= py < height, f"point: {point} - image shape: {image.shape}"
                    # px, py, visible
                    label.extend([format(px / width, ".6f"), format(py / height, ".6f"), 2.0])
                elif visible == 1:
                    assert 0 <= px < width and 0 <= py < height, f"point: {point} - image shape: {image.shape}"
                    label.extend([format(px / width, ".6f"), format(py / height, ".6f"), 1.0])
                else:
                    label.extend([0, 0, 0])

            labels.append(label)

        image_name = os.path.basename(image_path)
        dst_img_path = os.path.join(dst_image_root, image_name)
        shutil.copy(image_path, dst_img_path)

        name = os.path.splitext(image_name)[0]
        dst_label_path = os.path.join(dst_label_root, f"{name}.txt")
        np.savetxt(dst_label_path, np.array(labels), delimiter=' ', fmt='%s')


def main():
    args = parse_args()
    img_root = args.image
    label_path = args.label

    assert os.path.exists(img_root), img_root
    assert os.path.exists(label_path), label_path
    print(f"Parse {label_path}")
    results = load_label(label_path)
    print(f"Processing {len(results)} images")

    # 打乱数据
    random.seed(42)  # 设置随机种子以保证结果可复现
    random.shuffle(results)
    # 计算分割点
    split_index = int(len(results) * 0.8)
    # 分割数据
    train_data = results[:split_index]
    val_data = results[split_index:]
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    dst_root = args.dst
    print(f"Save to {dst_root}")
    create_dataset(train_data, img_root, dst_root, is_train=True)
    create_dataset(val_data, img_root, dst_root, is_train=False)


if __name__ == '__main__':
    main()
