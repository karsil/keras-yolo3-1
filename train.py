#! /usr/bin/env python

import argparse
import os
import numpy as np
import json
from voc import parse_voc_annotation
from yolo import create_yolov3_model, dummy_loss
from generator import BatchGenerator
from utils.utils import normalize, evaluate, makedirs
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from callbacks import CustomModelCheckpoint, CustomTensorBoard
from utils.multi_gpu_model import multi_gpu_model
import tensorflow as tf
import keras
from tqdm.keras import TqdmCallback
from keras.models import load_model
import logging
import shutil


config = tf.compat.v1.ConfigProto(
    allow_soft_placement=True,
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
    # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

def create_training_instances(
    train_annot_folder,
    train_image_folder,
    train_cache,
    valid_annot_folder,
    valid_image_folder,
    valid_cache,
    labels,
    train_val_split
):
    # parse annotations of the training set
    train_ints, train_labels = parse_voc_annotation(train_annot_folder, train_image_folder, train_cache, labels)

    # parse annotations of the validation set, if any, otherwise split the training set
    if os.path.exists(valid_annot_folder):
        valid_ints, valid_labels = parse_voc_annotation(valid_annot_folder, valid_image_folder, valid_cache, labels)
    else:
        print("valid_annot_folder not exists. Spliting the trainining set.")

        train_valid_split = int(train_val_split * len(train_ints))
        np.random.seed(0)
        np.random.shuffle(train_ints)
        np.random.seed()

        valid_ints = train_ints[train_valid_split:]
        train_ints = train_ints[:train_valid_split]

    # compare the seen labels with the given labels in config.json
    if len(labels) > 0:
        overlap_labels = set(labels).intersection(set(train_labels.keys()))

        print('Seen labels: \t'  + str(train_labels) + '\n')
        print('Given labels: \t' + str(labels))

        # return None, None, None if some given label is not in the dataset
        if len(overlap_labels) < len(labels):
            print('Some labels have no annotations! Please revise the list of labels in the config.json.')
            return None, None, None
    else:
        print('No labels are provided. Train on all seen labels.')
        print(train_labels)
        labels = train_labels.keys()

    max_box_per_image = max([len(inst['object']) for inst in (train_ints + valid_ints)])

    return train_ints, valid_ints, sorted(labels), max_box_per_image

def create_callbacks(saved_weights_name, tensorboard_logs, model_to_save):
    makedirs(tensorboard_logs)
    
    early_stop = EarlyStopping(
        monitor     = 'val_loss', 
        min_delta   = 0.01, 
        patience    = 5, 
        mode        = 'min', 
        verbose     = 1,
        restore_best_weights=True
    )
    checkpoint = CustomModelCheckpoint(
        model_to_save   = model_to_save,
        filepath        = saved_weights_name,# + '{epoch:02d}.h5', 
        monitor         = 'val_loss', 
        verbose         = 1, 
        save_best_only  = True, 
        mode            = 'min', 
        period          = 1
    )
    reduce_on_plateau = ReduceLROnPlateau(
        monitor  = 'val_loss',
        factor   = 0.2,
        patience = 3,
        verbose  = 1,
        mode     = 'min',
        epsilon  = 0.01,
        cooldown = 3,
        min_lr   = 0.0000001
    )
    tensorboard = CustomTensorBoard(
        log_dir                = tensorboard_logs,
        write_graph            = True,
        write_images           = True,
    )    

    nan = tf.keras.callbacks.TerminateOnNaN()

    return [early_stop, checkpoint, reduce_on_plateau, tensorboard, nan]

def create_model(
    nb_class, 
    anchors, 
    max_box_per_image, 
    max_grid, batch_size, 
    warmup_batches, 
    ignore_thresh, 
    multi_gpu, 
    saved_weights_name, 
    lr,
    grid_scales,
    obj_scale,
    noobj_scale,
    xywh_scale,
    class_scale,
    optimizer = None
):
    if multi_gpu > 1:
        logging.info(f"Create model for {multi_gpu} gpus")
        with tf.device('/cpu:0'):
            template_model, infer_model = create_yolov3_model(
                nb_class            = nb_class, 
                anchors             = anchors, 
                max_box_per_image   = max_box_per_image, 
                max_grid            = max_grid, 
                batch_size          = batch_size//multi_gpu, 
                warmup_batches      = warmup_batches,
                ignore_thresh       = ignore_thresh,
                grid_scales         = grid_scales,
                obj_scale           = obj_scale,
                noobj_scale         = noobj_scale,
                xywh_scale          = xywh_scale,
                class_scale         = class_scale
            )
    else:
        template_model, infer_model = create_yolov3_model(
            nb_class            = nb_class, 
            anchors             = anchors, 
            max_box_per_image   = max_box_per_image, 
            max_grid            = max_grid, 
            batch_size          = batch_size, 
            warmup_batches      = warmup_batches,
            ignore_thresh       = ignore_thresh,
            grid_scales         = grid_scales,
            obj_scale           = obj_scale,
            noobj_scale         = noobj_scale,
            xywh_scale          = xywh_scale,
            class_scale         = class_scale
        )  

    # load the pretrained weight if exists, otherwise load the backend weight only
    if os.path.exists(saved_weights_name): 
        logging.info("Loading pretrained weights.")
        template_model.load_weights(saved_weights_name)
    else:
        logging.info("Loading default backend weights.")
        template_model.load_weights("backend.h5", by_name=True)       

    if multi_gpu > 1:
        train_model = multi_gpu_model(template_model, gpus=multi_gpu)
    else:
        train_model = template_model      

    if not optimizer:
        optimizer = Adam(lr=lr, clipnorm=0.001)
    train_model.compile(loss=dummy_loss, optimizer=optimizer)             

    return train_model, infer_model

def _main_(args):
    config_path = args.conf
    logging.info("Loading config")

    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    ###############################
    #   Parse the annotations 
    ###############################
    logging.info("Create training instances")

    train_ints, valid_ints, labels, max_box_per_image = create_training_instances(
        config['train']['train_annot_folder'],
        config['train']['train_image_folder'],
        config['train']['cache_name'],
        config['valid']['valid_annot_folder'],
        config['valid']['valid_image_folder'],
        config['valid']['cache_name'],
        config['model']['labels'],
        config['train']['train_val_split']
    )
    print('\nTraining on: \t' + str(labels) + '\n')

    ###############################
    #   Create the generators 
    ###############################    
    logging.info("Create training generator")    
    train_generator = BatchGenerator(
        instances           = train_ints, 
        anchors             = config['model']['anchors'],   
        labels              = labels,        
        downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image   = max_box_per_image,
        batch_size          = config['train']['batch_size'],
        min_net_size        = config['model']['min_input_size'],
        max_net_size        = config['model']['max_input_size'],   
        shuffle             = True, 
        jitter              = 0.3, 
        norm                = normalize
    )
    logging.info("Create validation generator")
    valid_generator = BatchGenerator(
        instances           = valid_ints, 
        anchors             = config['model']['anchors'],   
        labels              = labels,        
        downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image   = max_box_per_image,
        batch_size          = config['train']['batch_size'],
        min_net_size        = config['model']['min_input_size'],
        max_net_size        = config['model']['max_input_size'],   
        shuffle             = True, 
        jitter              = 0.0, 
        norm                = normalize
    )

    ###############################
    #   Create the model 
    ###############################
    if os.path.exists(config['train']['saved_weights_name']): 
        config['train']['warmup_epochs'] = 0
    warmup_batches = config['train']['warmup_epochs'] * (config['train']['train_times']*len(train_generator))   

    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    multi_gpu = len(config['train']['gpus'].split(','))

    logging.info("Create model")
    train_model, infer_model = create_model(
        nb_class            = len(labels), 
        anchors             = config['model']['anchors'], 
        max_box_per_image   = max_box_per_image, 
        max_grid            = [config['model']['max_input_size'], config['model']['max_input_size']], 
        batch_size          = config['train']['batch_size'], 
        warmup_batches      = warmup_batches,
        ignore_thresh       = config['train']['ignore_thresh'],
        multi_gpu           = multi_gpu,
        saved_weights_name  = config['train']['saved_weights_name'],
        lr                  = config['train']['learning_rate'],
        grid_scales         = config['train']['grid_scales'],
        obj_scale           = config['train']['obj_scale'],
        noobj_scale         = config['train']['noobj_scale'],
        xywh_scale          = config['train']['xywh_scale'],
        class_scale         = config['train']['class_scale'],
    )

    ###############################
    #   Kick off the training
    ###############################
    callbacks = create_callbacks(config['train']['saved_weights_name'], config['train']['tensorboard_dir'], infer_model)
    callbacks = callbacks + [
        TqdmCallback(batch_size=config['train']['batch_size'])
    ]
    logging.info("Start training")
    train_model.fit_generator(
        generator        = train_generator, 
        steps_per_epoch  = len(train_generator) * config['train']['train_times'], 
        epochs           = config['train']['nb_epochs'] + config['train']['warmup_epochs'], 
        verbose          = 2 if config['train']['debug'] else 1,
        callbacks        = callbacks, 
        validation_data  = valid_generator,
        workers          = 4,
        max_queue_size   = 8
    )

    # make a GPU version of infer_model for evaluation
    if multi_gpu > 1:
        infer_model = load_model(config['train']['saved_weights_name'])

    ###############################
    #   Run the evaluation
    ###############################   
    logging.info("Start evaluation")
    # compute mAP for all the classes
    average_precisions, _, _ = evaluate(
        model=infer_model, 
        generator=valid_generator,
        labels=labels,
        save_path=config['train']['result_dir']
        )

    # print the score
    for label, average_precision in average_precisions.items():
        print(labels[label] + ': {:.4f}'.format(average_precision))
    print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))           

    src_dir = os.path.abspath(config['train']['saved_weights_name'])
    target_dir = os.path.join(
        os.getcwd(),
        config['train']['result_dir'],
        config['train']['saved_weights_name']
    )
    print(f"Moving weights from {src_dir} to {target_dir}")
    shutil.move(src_dir,target_dir)

    src_dir = os.path.abspath(config['train']['tensorboard_dir'])
    target_dir = os.path.join(
        os.getcwd(),
        config['train']['result_dir'],
        config['train']['tensorboard_dir']
    )
    print(f"Moving logs from {src_dir} to {target_dir}")
    shutil.move(src_dir,target_dir)


if __name__ == '__main__':
    logging.basicConfig(filename='train.log', level=logging.DEBUG)
    argparser = argparse.ArgumentParser(description='train and evaluate YOLO_v3 model on any dataset')
    argparser.add_argument('-c', '--conf', default='config.json', help='path to configuration file')   

    args = argparser.parse_args()
    _main_(args)
