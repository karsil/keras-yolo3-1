#! /usr/bin/env python

import argparse
import os
import numpy as np
import json
import csv
from pathlib import Path
from voc import parse_voc_annotation
from yolo import create_yolov3_model
from generator import BatchGenerator
from utils.utils import normalize, evaluate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.models import load_model

def _main_(args):
    config_path = args.conf

    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    ###############################
    #   Create the validation generator
    ###############################  
    valid_ints, labels = parse_voc_annotation(
        config['valid']['valid_annot_folder'], 
        config['valid']['valid_image_folder'], 
        config['valid']['cache_name'],
        config['model']['labels']
    )

    labels = labels.keys() if len(config['model']['labels']) == 0 else config['model']['labels']
    labels = sorted(labels)
   
    valid_generator = BatchGenerator(
        instances           = valid_ints, 
        anchors             = config['model']['anchors'],   
        labels              = labels,        
        downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image   = 0,
        batch_size          = config['train']['batch_size'],
        min_net_size        = config['model']['min_input_size'],
        max_net_size        = config['model']['max_input_size'],   
        shuffle             = True, 
        jitter              = 0.0, 
        norm                = normalize
    )

    ###############################
    #   Load the model and do evaluation
    ###############################
    #os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    #model_path = "/home/jsteeg/keras-yolo3-1/studies/20200923-192151-996072/trained_ufo.h5"
    #infer_model = load_model(model_path)
    infer_model = load_model(config['train']['saved_weights_name_eval'])

    result_dir = result_dir = os.path.join(os.path.dirname(config['train']['saved_weights_name_eval']), "eval")
    if args.dir:
        result_dir = args.dir

    ious = np.arange(0.05, 1, 0.05)
    #ious = np.arange(0.7, 1, 0.1)
    print(f"Saving results to {result_dir}")

    results = {}
    for iou in ious:
        iou = round(iou, 2)
        print(f"Processing IoU@{iou}...")
        save_path = os.path.join(result_dir, str(iou))
        Path(save_path).mkdir(parents=True, exist_ok=True)
        # compute mAP for all the classes
        average_precisions, average_f1s, class_weights = evaluate(model=infer_model, generator=valid_generator,iou_threshold=iou, labels=labels, save_path=save_path)

        avg_p = 0.0
        avg_f1 = 0.0
        
        # AFAIK, map is calculated with no respect to class sizes
        use_class_weights = False
        if use_class_weights:
            for avgP, f1, weight in zip(list(average_precisions.values()), list(average_f1s.values()), list(class_weights.values())):
                avg_p += weight * avgP
                avg_f1 += weight * f1
        else:
            avg_p = np.mean(list(average_precisions.values()))
            avg_f1 = np.mean(list(average_f1s.values()))

        results[iou] = {
            "ap": avg_p,
            "f1": avg_f1
        }

    ap_results = os.path.join(result_dir, 'eval_result.csv')
    with open(ap_results, 'w') as csvfile:
        filewriter = csv.writer(csvfile)
        filewriter.writerow(['AP@', 'Score', "f1"])

        for iou, metrics in results.items():
            i = "{:.2f}".format(iou)
            ap = "{:.4f}".format(metrics["ap"])
            f1 = "{:.4f}".format(metrics["f1"])
            filewriter.writerow([i, ap, f1])
            print('AP@{}: {} - F1: {}'.format(i, ap, f1))

    map_result = os.path.join(result_dir, 'map.txt')
    map_value = sum([value["ap"] for value in results.values()]) / len(results)
    with open(map_result, 'w') as map_file:
        map_file.write("mAP: {:.4f}".format(map_value))
    print('mAP: {:.4f}'.format(map_value))
        

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Evaluate YOLO_v3 model on any dataset')
    argparser.add_argument('-c', '--conf', default='config.json', help='path to configuration file')
    argparser.add_argument('-d', '--dir', default=None, type=dir_path, help='Path to store results. By default a generic folder beside the weight will be used')
    args = argparser.parse_args()
    _main_(args)
