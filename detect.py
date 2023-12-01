import os
import sys

import torch
from torch.utils.data import DataLoader
from ultralytics import YOLO
from yolov5.utils.general import coco80_to_coco91_class

module_path = os.path.abspath(os.getcwd() + "/src")
if module_path not in sys.path:
    sys.path.append(module_path)

from config_parser import analyze_config, config, get_r_high_dir, get_r_low_dir
from dataset import PathDataset
from utility.path_utils import get_last_path_element


def load_model():
    detector = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    # yolov8 = YOLO('yolov8n.pt', map_location=config.device)

    detector.iou = 0
    detector.conf = config.score_threshold

    return detector


def save_results(detector, classes, filepaths, save_dir, save_demo=True):
    detect_results = detector(filepaths)
    
    # Save demo detected
    if save_demo:
        detect_results.save(labels=True, save_dir=save_dir, exist_ok=True)

    # Save bounding boxes
    for i, xywh in enumerate(detect_results.xywh):
        xywh = xywh.numpy()
        filename = get_last_path_element(filepaths[i]).split('.')[0]

        with open(os.path.join(save_dir, f"{filename}.txt"), "w+") as f:
            for box in xywh:
                box[5] = classes[int(box[5])] # Map COCO 80 classes to COCO 90 classes
                f.write(' '.join(map(str, box)) + '\n')



def detect(detector, classes, input_path, output_path=None):
    if not output_path:
        output_path = input_path
    
    dataset = PathDataset(input_path, config.image_extensions)
    dataloader = DataLoader(dataset=dataset, num_workers=2, batch_size=config.batch_size, shuffle=True)
    detect_dir = os.path.join(output_path, 'detects')

    for i, images_paths in enumerate(dataloader):
        save_demo = i < config.demo_count
        save_results(detector, classes, images_paths, detect_dir, save_demo)



if __name__ == "__main__":
    analyze_config(os.path.abspath('./config.ini'))
    
    detector = load_model()
    classes = coco80_to_coco91_class()

    detect(detector, classes, config.input_dir, config.output_dir)
    for r in config.r_values:
        for low in [True, False]:
            if low:
                save_dir = get_r_low_dir(r)
            else:
                save_dir = get_r_high_dir(r)

        detect(detector, classes, save_dir)
