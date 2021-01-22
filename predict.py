#! /usr/bin/env python

import tensorflow as tf
import os
from pathlib import Path
import argparse
import json
import cv2
from utils.utils import get_yolo_boxes, makedirs
from utils.bbox import draw_boxes, write_detection_results
from keras.models import load_model
from tqdm.auto import tqdm
import numpy as np

def _main_(args):
    config_path  = Path(args.conf)
    input_path   = Path(args.input)
    output_path  = Path(args.output)

    assert config_path.exists() and input_path.exists()

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    makedirs(output_path)
    if not args.text_only:
        image_output_path = output_path.joinpath('images')
        if not image_output_path.exists(): image_output_path.mkdir()
    text_output_path = output_path.joinpath('detections')
    if not text_output_path.exists(): text_output_path.mkdir()

    ###############################
    #   Set some parameter
    ###############################       
    net_h, net_w = 416, 416 # a multiple of 32, the smaller the faster
    obj_thresh, nms_thresh = 0.5, 0.45

    ###############################
    #   Load the model
    ###############################
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    infer_model = load_model(config['train']['saved_weights_name'])

    ###############################
    #   Predict bounding boxes 
    ###############################
    if 'webcam' in str(input_path): # do detection on the first webcam
        video_reader = cv2.VideoCapture(0)

        # the main loop
        batch_size  = 1
        images      = []
        while True:
            ret_val, image = video_reader.read()
            if ret_val == True: images += [image]

            if (len(images)==batch_size) or (ret_val==False and len(images)>0):
                batch_boxes = get_yolo_boxes(infer_model, images, net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)

                for i in range(len(images)):
                    draw_boxes(images[i], batch_boxes[i], config['model']['labels'], obj_thresh) 
                    cv2.imshow('video with bboxes', images[i])
                images = []
            if cv2.waitKey(1) == 27: 
                break  # esc to quit
        cv2.destroyAllWindows()        
    elif input_path.suffix == '.mp4': # do detection on a video  
        video_out = image_output_path.joinpath(input_path.name)
        video_reader = cv2.VideoCapture(input_path)

        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

        video_writer = cv2.VideoWriter(str(video_out),
                               cv2.VideoWriter_fourcc(*'MPEG'), 
                               50.0, 
                               (frame_w, frame_h))
        # the main loop
        batch_size  = 1
        images      = []
        start_point = 0 #%
        show_window = False
        for i in tqdm(range(nb_frames)):
            _, image = video_reader.read()

            if (float(i+1)/nb_frames) > start_point/100.:
                images += [image]

                if (i%batch_size == 0) or (i == (nb_frames-1) and len(images) > 0):
                    # predict the bounding boxes
                    batch_boxes = get_yolo_boxes(infer_model, images, net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)

                    for i in range(len(images)):
                        # draw bounding boxes on the image using labels
                        draw_boxes(images[i], batch_boxes[i], config['model']['labels'], obj_thresh)   

                        # show the video with detection bounding boxes          
                        if show_window: cv2.imshow('video with bboxes', images[i])  

                        # write result to the output video
                        video_writer.write(images[i]) 
                    images = []
                if show_window and cv2.waitKey(1) == 27: break  # esc to quit

        if show_window: cv2.destroyAllWindows()
        video_reader.release()
        video_writer.release()       
    else: # do detection on an image or a set of images
        image_paths = []

        # read all images in given folder
        if input_path.is_dir(): 
            for inp_file in input_path.glob('*.*'):
                image_paths += [inp_file]
        # read image paths from a file list
        elif input_path.suffix == '.txt':
            for img_path in  input_path.read_text().splitlines():
                img_path = Path(img_path)
                # works with relative and absolute paths
                if not img_path.is_absolute():
                    img_path = input_path.parent.joinpath(img_path)

                assert img_path.exists(), f"{img_path}"
                image_paths += [img_path]
        # use a single image
        else:
            image_paths += [input_path]

        image_paths = [inp_file for inp_file in image_paths if (inp_file.suffix in ['.jpg', '.png', 'JPEG'])]

        # the main loop
        for image_path in tqdm(image_paths):
            
            image = cv2.imread(str(image_path))

            # predict the bounding boxes
            boxes = get_yolo_boxes(infer_model, [image], net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)[0]

            # draw bounding boxes on the image using labels
            labels = sorted(config['model']['labels'])

            write_detection_results(boxes, labels, txt_output_file=str(text_output_path.joinpath(Path(image_path).stem+'.txt')))

            # write the image with bounding boxes to file
            if not args.text_only:
                draw_boxes(image, boxes, labels, obj_thresh) 
                cv2.imwrite(str(image_output_path.joinpath(image_path.name)), np.uint8(image))         

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    argparser.add_argument('-i', '--input', help='path to a single image, a directory of images, a text file of image paths, a video or webcam')    
    argparser.add_argument('-o', '--output', default='output/', help='path to output directory')
    argparser.add_argument('--text_only', action='store_true', help='only output the result-coordinates in text files (no images).')      
    
    args = argparser.parse_args()
    _main_(args)
