from pathlib import Path
import os
from typing import List
import shutil
from abc import ABC
from tqdm import tqdm
import argparse
import json
import logging
import cv2
from utils.bbox import draw_boxes
from keras.models import load_model
from utils.utils import get_yolo_boxes
import numpy as np

UFO_CLASSES = None


class BBox(ABC):
	pass


class Detection(BBox):
	def __init__(self, xmin: int, ymin: int, xmax: int, ymax: int, class_name: str, score: float):
			self.xmin = xmin,
			self.ymin = ymin
			self.xmax = xmax
			self.ymax = ymax
			self.class_name = class_name
			self.score = score
	
	def __str__(self):
		# fish_cod 0.05917061120271683 9 6 79 410
		# TODO Why the hell does this one not work?
		xmin = str(self.xmin).replace(",", '').replace("(", '').replace(")", '')
		return f"{self.class_name} {str(self.score)} {str(xmin)} {str(self.ymin)} {str(self.xmax)} {str(self.ymax)}"


class UFOAnnotation(BBox):
	def __init__(self, xmin: int, ymin: int, xmax: int, ymax: int, class_id: int):
		self.xmin = xmin,
		self.ymin = ymin
		self.xmax = xmax
		self.ymax = ymax
		self.class_id = class_id
		self.class_name = UFO_CLASSES[class_id]

		#print(f"Create: {str(self)} -> {self.class_name}")

	def __str__(self):
		# goal: e.g. "fish_cod 109 0 195 69"
		# TODO Why the hell does this one not work?
		xmin = str(self.xmin).replace(",", '').replace("(", '').replace(")", '')
		return f"{self.class_name} {xmin} {str(self.ymin)} {str(self.xmax)} {str(self.ymax)}"


class Sample:
	def __init__(self, path: str, objects: List[BBox]):
		self.path = path
		self.objects = objects

	def save_in_folder(self, dir_path: Path):
		dst = dir_path.joinpath(self.path).with_suffix('.txt')
		#print(f"Store to {dst}")
		dst.touch()
		m = "\n".join(list(map(str, self.objects)))
		dst.write_text(m)


def parse_and_store_groundtruth(target_dir: Path, image_path: Path, annotations: List[UFOAnnotation]) -> None:
	sample = Sample(image_path.name, annotations)
	sample.save_in_folder(target_dir)


def parse_from_yolo(line: str) -> (Path, List[UFOAnnotation]):
	data = line.split()
	filepath = Path(data[0])
	annots = []
	for gt in data[1:]:
		gt = gt.split(",")
		gt_int = list(map(int, gt))
		annots.append(UFOAnnotation(gt_int[0], gt_int[1], gt_int[2], gt_int[3], gt_int[4]))
	return filepath, annots


def setup(args):
    config_path = args.conf
    logging.info("Loading config")

    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    global UFO_CLASSES
    UFO_CLASSES = config['model']['labels']
    return config


def validate(config, weights_path: str, dataset_file: str, output_dir: str, save_output: bool = False):
    with open(dataset_file, 'r') as f:
        lines = f.readlines()
        
    dst = Path(output_dir)
    if dst.is_dir():
        shutil.rmtree(dst)
    dst.mkdir()

    gt_dir = dst.joinpath("groundtruth")
    gt_dir.mkdir()

    detect_dir = dst.joinpath("detection_results")
    detect_dir.mkdir()

    if save_output:
        bbox_dir = dst.joinpath("bbox_images")
        bbox_dir.mkdir()


    net_h, net_w = 416, 416 # a multiple of 32, the smaller the faster
    obj_thresh, nms_thresh = 0.05, 0.45

    #os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    model = load_model(weights_path)

    for l in tqdm(lines):
        filepath, annots = parse_from_yolo(line=l)
        assert filepath.is_file()

        # handle groundtruth
        parse_and_store_groundtruth(target_dir=gt_dir, image_path=filepath, annotations=annots)

        # handle inference
        img = cv2.imread(str(filepath))

        det_boxes = get_yolo_boxes(model, [img], net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)[0]

        bboxes = [
        	Detection(xmin=b.xmin, ymin=b.ymin, xmax=b.xmax, ymax=b.ymax, class_name=UFO_CLASSES[b.get_label()], score=b.get_score()) for b in det_boxes if b.get_score() > 0.0
        ]

        sample = Sample(filepath.name, bboxes)
        sample.save_in_folder(detect_dir)

        if save_output:
            img_out = os.path.join(str(bbox_dir), filepath.name)
            draw_boxes(img, det_boxes, config['model']['labels'], obj_thresh)
            cv2.imwrite(img_out, np.uint8(img))


def main(args):
    config = setup(args)

    weights_path = "/home/jsteeg/keras-yolo3-1/training_02_21/trained_ufo.h5"
    dataset_file = "/home/jsteeg/ufo_data/yolo_no_crop/test_dataset.txt"
    output_dir = "tmp_validate_results"
    save_output = True
    validate(config=config, weights_path=weights_path, dataset_file=dataset_file, output_dir=output_dir, save_output=save_output)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    argparser = argparse.ArgumentParser(description='train and evaluate YOLO_v3 model on any dataset')
    argparser.add_argument('-c', '--conf', default='config.json', help='path to configuration file')   

    args = argparser.parse_args()
    main(args)
