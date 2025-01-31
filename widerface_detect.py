# -*- coding: utf-8 -*-

"""
@Time    : 2025/1/26 11:44
@File    : widerface_detect.py
@Author  : zj
@Description:

Usage - predict using YOLOv5:
    $ python widerface_detect.py --model yolov5nu_widerface.pt --source ../datasets/widerface/images/val/ --folder_pict ../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt --save_txt true --conf 0.001 --iou 0.7 --max_det 300 --batch 1 --device 0
    $ python widerface_detect.py --model yolov8n_widerface.pt --source ../datasets/widerface/images/val/ --folder_pict ../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt --save_txt true --conf 0.001 --iou 0.7 --max_det 300 --batch 1 --device 0

Usage - predict using YOLOv5:
    $ python widerface_detect.py --model yolov8n_widerface.pt --source ../datasets/widerface/images/val/ --folder_pict ../datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt --save_txt true --conf 0.001 --iou 0.7 --max_det 300 --batch 1 --device 0

"""
import os
import re
import cv2
import torch

from pathlib import Path

import yolo8face_utils
from ultralytics.utils import DEFAULT_CFG, LOGGER, ops, colorstr
from yolo8face_utils import parse_args

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect import DetectionPredictor


class WiderFaceResults(Results):

    def __init__(self, orig_img, path, names, boxes=None, masks=None, probs=None, keypoints=None, obb=None,
                 speed=None) -> None:
        super().__init__(orig_img, path, names, boxes, masks, probs, keypoints, obb, speed)

    def save_txt(self, txt_file, save_conf=False):
        # return super().save_txt(txt_file, save_conf)
        """
        Save detection results to a text file.

        Args:
            txt_file (str | Path): Path to the output text file.
            save_conf (bool): Whether to include confidence scores in the output.

        Returns:
            (str): Path to the saved text file.

        Examples:
            >>> from ultralytics import YOLO
            >>> model = YOLO("yolov8n.pt")
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            ...     result.save_txt("output.txt")

        Notes:
            - The file will contain one line per detection or classification with the following structure:
              - For detections: `class confidence x_center y_center width height`
              - For classifications: `confidence class_name`
              - For masks and keypoints, the specific formats will vary accordingly.
            - The function will create the output directory if it does not exist.
            - If save_conf is False, the confidence scores will be excluded from the output.
            - Existing contents of the file will not be overwritten; new results will be appended.
        """
        is_obb = self.obb is not None
        boxes = self.obb if is_obb else self.boxes
        masks = self.masks
        probs = self.probs
        kpts = self.keypoints
        texts = []
        if probs is not None:
            # Classify
            [texts.append(f"{probs.data[j]:.2f} {self.names[j]}") for j in probs.top5]
        elif boxes:
            # Detect/segment/pose
            for j, d in enumerate(boxes):
                c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())
                # [x1, y1, x2, y2] -> [x1, y1, w, h]
                # line = (c, *(d.xyxy.view(-1) if is_obb else d.xywhn.view(-1)))
                x1, y1, x2, y2 = d.xyxy.view(-1).int()
                w, h = x2 - x1, y2 - y1
                line = (x1, y1, w, h, conf)
                if masks:
                    seg = masks[j].xyn[0].copy().reshape(-1)  # reversed mask.xyn, (n,2) to (n*2)
                    line = (c, *seg)
                if kpts is not None:
                    kpt = torch.cat((kpts[j].xyn, kpts[j].conf[..., None]), 2) if kpts[j].has_visible else kpts[j].xyn
                    line += (*kpt.reshape(-1).tolist(),)
                line += (conf,) * save_conf + (() if id is None else (id,))
                texts.append(("%g " * len(line)).rstrip() % line)

        Path(txt_file).parent.mkdir(parents=True, exist_ok=True)  # make directory
        with open(txt_file, 'w') as f:
            f.write(f"{Path(self.path).stem}\n")
            f.write(f"{len(texts)}\n")

        if texts:
            # Path(txt_file).parent.mkdir(parents=True, exist_ok=True)  # make directory
            with open(txt_file, "a") as f:
                f.writelines(text + "\n" for text in texts)


class WiderFacePredictor(DetectionPredictor):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None, folder_pict=None):
        super().__init__(cfg, overrides, _callbacks)
        assert folder_pict is not None

        # read folder_pict
        pict_folder = {}
        assert os.path.isfile(folder_pict), folder_pict
        with open(folder_pict, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line == "":
                    continue
                if not line.endswith('.jpg'):
                    continue
                line = line.strip().split('/')
                pict_folder[line[-1]] = line[-2]
        print(pict_folder.__len__())
        self.pict_folder = pict_folder

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            # results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
            results.append(WiderFaceResults(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

    def stream_inference(self, source=None, model=None, *args, **kwargs):
        # return super().stream_inference(source, model, *args, **kwargs)
        """Streams real-time inference on camera feed and saves results to file."""
        if self.args.verbose:
            LOGGER.info("")

        # Setup model
        if not self.model:
            self.setup_model(model)

        with self._lock:  # for thread-safe inference
            # Setup source every time predict is called
            self.setup_source(source if source is not None else self.args.source)

            # Check if save_dir/ label file exists
            if self.args.save or self.args.save_txt:
                # (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
                (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

            # Warmup model
            if not self.done_warmup:
                self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
                self.done_warmup = True

            self.seen, self.windows, self.batch = 0, [], None
            profilers = (
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
            )
            self.run_callbacks("on_predict_start")
            for self.batch in self.dataset:
                self.run_callbacks("on_predict_batch_start")
                paths, im0s, s = self.batch
                # LOGGER.info("\n".join(s))
                # LOGGER.info("\nadfadfasdfadsf")

                # Preprocess
                with profilers[0]:
                    im = self.preprocess(im0s)

                # Inference
                with profilers[1]:
                    preds = self.inference(im, *args, **kwargs)
                    if self.args.embed:
                        yield from [preds] if isinstance(preds, torch.Tensor) else preds  # yield embedding tensors
                        continue

                # Postprocess
                with profilers[2]:
                    self.results = self.postprocess(preds, im, im0s)
                self.run_callbacks("on_predict_postprocess_end")

                # Visualize, save, write results
                n = len(im0s)
                for i in range(n):
                    self.seen += 1
                    self.results[i].speed = {
                        "preprocess": profilers[0].dt * 1e3 / n,
                        "inference": profilers[1].dt * 1e3 / n,
                        "postprocess": profilers[2].dt * 1e3 / n,
                    }
                    if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                        s[i] += self.write_results(i, Path(paths[i]), im, s)

                # Print batch results
                if self.args.verbose:
                    LOGGER.info("\n".join(s))

                self.run_callbacks("on_predict_batch_end")
                yield from self.results

        # Release assets
        for v in self.vid_writer.values():
            if isinstance(v, cv2.VideoWriter):
                v.release()

        # Print final results
        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1e3 for x in profilers)  # speeds per image
            LOGGER.info(
                f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape "
                f"{(min(self.args.batch, self.seen), 3, *im.shape[2:])}" % t
            )
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob("labels/*.txt")))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ""
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
        self.run_callbacks("on_predict_end")

    def write_results(self, i, p, im, s):
        # return super().write_results(i, p, im, s)
        """Write inference results to a file or directory."""
        string = ""  # print string
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        if self.source_type.stream or self.source_type.from_img or self.source_type.tensor:  # batch_size >= 1
            string += f"{i}: "
            frame = self.dataset.count
        else:
            match = re.search(r"frame (\d+)/", s[i])
            frame = int(match[1]) if match else None  # 0 if frame undetermined

        # self.txt_path = self.save_dir / "labels" / (p.stem + ("" if self.dataset.mode == "image" else f"_{frame}"))
        self.txt_path = self.save_dir / "labels" / f"{self.pict_folder[p.name]}" / (
                p.stem + ("" if self.dataset.mode == "image" else f"_{frame}"))
        # LOGGER.info(f"txt_path: {self.txt_path}")
        string += "{:g}x{:g} ".format(*im.shape[2:])
        result = self.results[i]
        result.save_dir = self.save_dir.__str__()  # used in other locations
        string += f"{result.verbose()}{result.speed['inference']:.1f}ms"

        # Add predictions to image
        if self.args.save or self.args.show:
            self.plotted_img = result.plot(
                line_width=self.args.line_width,
                boxes=self.args.show_boxes,
                conf=self.args.show_conf,
                labels=self.args.show_labels,
                im_gpu=None if self.args.retina_masks else im[i],
            )

        # Save results
        if self.args.save_txt:
            result.save_txt(f"{self.txt_path}.txt", save_conf=self.args.save_conf)
        if self.args.save_crop:
            result.save_crop(save_dir=self.save_dir / "crops", file_name=self.txt_path.stem)
        if self.args.show:
            self.show(str(p))
        if self.args.save:
            self.save_predicted_images(str(self.save_dir / p.name), frame)

        return string


def main():
    overrides = parse_args(folder_pict=True)
    assert overrides['model'] is not None, 'model must be specified'
    assert overrides['source'] is not None, 'source must be specified'
    assert overrides['folder_pict'] is not None, 'folder_pict must be specified'

    overrides['mode'] = 'predict'
    print(overrides)
    folder_pict = overrides['folder_pict']
    overrides.pop('folder_pict')

    # 初始化预测器并开始预测
    predictor = WiderFacePredictor(overrides=overrides, folder_pict=folder_pict)
    predictor.predict_cli()


if __name__ == "__main__":
    main()
