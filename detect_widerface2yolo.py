# -*- coding: utf-8 -*-

"""
@Time    : 2024/6/16 20:09
@File    : widerface2yolo.py
@Author  : zj
@Description:

Usage - Convert the WIDERFACE dataset format to YOLO:
    $ python3 widerface2yolo.py ../datasets/widerface/WIDER_train/images ../datasets/widerface/wider_face_split/wider_face_train_bbx_gt.txt ../datasets/widerface/
    $ python3 widerface2yolo.py ../datasets/widerface/WIDER_val/images ../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt ../datasets/widerface/

The original dataset format is as follows:

```text
├── WIDER_train
│   └── images
        ├── 0--Parade
        ├── 10--People_Marching
        ├── 11--Meeting
        ├── 12--Group
        ├── 13--Interview
        ├── 14--Traffic
        ├── ...
        ├── ...
├── WIDER_val
│   └── images
├── wider_face_split
│   ├── wider_face_train_bbx_gt.txt
│   ├── wider_face_val_bbx_gt.txt
```

After processing, the data format is as follows:

```text
├── images
│   ├── train
│   └── val
├── labels
│   ├── train
│   ├── val
```

"""

import os
import cv2
import shutil
import argparse

import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="WiderFace2YOLO")
    parser.add_argument('image', metavar='IMAGE', type=str, help='WiderFace image root.')
    parser.add_argument('label', metavar='LABEL', type=str, help='WiderFace label path.')

    parser.add_argument('dst', metavar='DST', type=str, help='YOLOLike data root.')

    args = parser.parse_args()
    print("args:", args)
    return args


def parse_label_file(label_file_path):
    results = []
    with open(label_file_path, 'r') as file:
        while True:
            # Read the image path
            image_path_line = file.readline().strip()
            if not image_path_line:
                break

            # Read the number of labels
            num_labels_line = file.readline().strip()
            num_labels = int(num_labels_line)
            if num_labels == 0:
                _ = file.readline().strip()
                continue

            # Read bounding boxes
            bounding_boxes = []
            for _ in range(num_labels):
                bbox_line = file.readline().strip()
                bbox_data = list(map(int, bbox_line.split()))
                bounding_boxes.append(bbox_data[:4])

            results.append({
                'image_path': image_path_line,
                'num_labels': num_labels,
                'bounding_boxes': bounding_boxes
            })

    return results


def main():
    args = parse_args()

    dst_root = args.dst
    img_root = args.image
    label_path = args.label

    if 'val' in label_path:
        dst_img_root = os.path.join(dst_root, "images/val")
        dst_label_root = os.path.join(dst_root, "labels/val")
    else:
        dst_img_root = os.path.join(dst_root, "images/train")
        dst_label_root = os.path.join(dst_root, "labels/train")
    if not os.path.exists(dst_img_root):
        os.makedirs(dst_img_root)
    if not os.path.exists(dst_label_root):
        os.makedirs(dst_label_root)

    cls_id = 0

    assert os.path.exists(img_root), img_root
    assert os.path.exists(label_path), label_path
    print(f"Parse {label_path}")
    results = parse_label_file(label_path)

    print(f"Processing {len(results)} images")
    for result in tqdm(results):
        image_path = os.path.join(img_root, result["image_path"])
        assert os.path.isfile(image_path), image_path

        image = cv2.imread(image_path)
        height, width, channels = image.shape

        labels = []
        bounding_boxes = result["bounding_boxes"]
        for box in bounding_boxes:
            x1, y1, box_w, box_h = box
            x_c = format(1.0 * (x1 + box_w / 2) / width, ".6f")
            y_c = format(1.0 * (y1 + box_h / 2) / height, ".6f")
            box_w = format(1.0 * box_w / width, ".6f")
            box_h = format(1.0 * box_h / height, ".6f")
            labels.append([int(cls_id), x_c, y_c, box_w, box_h])

        image_name = os.path.basename(image_path)
        dst_img_path = os.path.join(dst_img_root, image_name)
        shutil.copy(image_path, dst_img_path)

        name = os.path.splitext(image_name)[0]
        dst_label_path = os.path.join(dst_label_root, f"{name}.txt")
        np.savetxt(dst_label_path, np.array(labels), delimiter=' ', fmt='%s')


if __name__ == '__main__':
    main()
