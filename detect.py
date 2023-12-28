import os
import sys
from logging import Logger
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from ultralytics import YOLO
from yolov5.utils.general import coco80_to_coco91_class

module_path = os.path.abspath(os.getcwd() + "/src")
if module_path not in sys.path:
    sys.path.append(module_path)

from config_parser import analyze_config, config
from dataset import PathDataset
from utility.mylogger import MyLogger
from utility.path_utils import (count_filepaths, create_path_if_not_exists,
                                get_exp_folders, get_last_path_element)


def load_model():
    detector = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    # yolov8 = YOLO('yolov8n.pt', map_location=config.device)

    detector.iou = 0
    detector.conf = config.score_threshold

    return detector


def save_results(detector, coco_91, filepaths, save_dir):
    detect_results = detector(filepaths)

    # Save bounding boxes
    for i, xyxy in enumerate(detect_results.xyxy):
        xyxy = xyxy.numpy()
        filename = get_last_path_element(filepaths[i]).split('.')[0]

        with open(Path(save_dir) / f"{filename}.txt", "w+") as f:
            for box in xyxy:
                box[5] = coco_91[int(box[5])] # Map COCO 80 classes to COCO 90 classes
                f.write(' '.join(map(str, box)) + '\n')



def detect(logger: Logger, detector, classes, input_path, output_path=None):
    if not output_path:
        output_path = input_path
    detect_dir = Path(output_path) / 'detects'

    dataset = PathDataset(input_path, config.image_extensions)

    if len(dataset) == 0:
        logger.warning(f"Skipping {input_path}: no image found in this directory")
        return

    if not config.force_detect and count_filepaths(detect_dir, ['txt']) == len(dataset):
        logger.info(f"Skipping {input_path}: force_detect option is False and this path got enough detect result files")
        return
    
    create_path_if_not_exists(detect_dir)

    logger.info(f"Detecting: {input_path}")
    try:
        dataloader = DataLoader(dataset=dataset, batch_size=config.batch_size)
    except ValueError:
        logger.error(f'Input path "{input_path}" does not contain any image with allowed extension.')
        return

    for i, images_paths in enumerate(dataloader):
        save_results(detector, classes, images_paths, detect_dir)



if __name__ == "__main__":
    analyze_config(os.path.abspath('./config.ini'))
    logger = MyLogger.getLog()
    
    detector = load_model()
    coco_91 = coco80_to_coco91_class()

    # detect input images
    detect(logger, detector, coco_91, config.input_dir, config.output_dir)

    # detect analyze low frequency images
    # for r in config.r_values:
    #     low_dir = get_r_low_dir(r)
    #     detect(logger, detector, coco_91, low_dir)

    # detect experiments images
    exp_folders = get_exp_folders(config.exp_dir)
    for folder in exp_folders:
        detect(logger, detector, coco_91, folder)
