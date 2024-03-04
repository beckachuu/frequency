import csv
import os
import sys
from pathlib import Path

try:
    import cupy as np
    print("Running cupy with GPU")
except ImportError:
    import numpy as np
    print("Running numpy with CPU")

module_path = os.path.abspath(os.getcwd() + "/src")
if module_path not in sys.path:
    sys.path.append(module_path)

from config_parser import analyze_config, config
from const.constants import coco_91_classes
from utility.map_util import (calculate_AP_for_class,
                              calculate_tp_fp_range_for_class,
                              calculate_tp_fp_range_from_path, get_detect_bbox,
                              load_detect_data, load_gt_data)
from utility.mylogger import MyLogger
from utility.path_utils import (get_exp_folders, get_filepaths_list,
                                get_last_path_element, read_dict_from_json)


def eval(exp_detect_folders, groundtruth, iou_list):

    gt_instances = [0] * len(coco_91_classes)
    origins_ap = [[0 for _ in range(len(iou_list))] for _ in range(len(coco_91_classes))]
    exp_APs = [[[0 for _ in range(len(iou_list))] for _ in range(len(coco_91_classes))] for _ in range(len(exp_detect_folders))]
    
    tp = [[[] for _ in range(len(iou_list))] for _ in range(len(coco_91_classes))]
    fp = [[[] for _ in range(len(iou_list))] for _ in range(len(coco_91_classes))]

    # # Calculate true positive, false positive for orginal images
    origins_paths = get_filepaths_list(Path(config.output_dir) / 'detects', ['txt'])
    for file_path in origins_paths:
        image_id = int(get_last_path_element(file_path).split('.')[0])
        gt_data = load_gt_data(groundtruth, image_id)
        detect_data = load_detect_data(get_detect_bbox(file_path))

        # accumulate_tp_fp for each class
        for class_ind in gt_data['labels']:
            tp_range, fp_range = calculate_tp_fp_range_for_class(gt_data, detect_data, class_ind)
            for i in range(len(iou_list)):
                tp[class_ind-1][i] += tp_range[i]
                fp[class_ind-1][i] += fp_range[i]
            
            gt_instances[class_ind-1] += gt_data['labels'].count(class_ind)

    # alternate way to calculate tp, fp (but no return for gt_instances)
    # tp, fp = calculate_tp_fp_range_from_path(Path(config.output_dir) / 'detects', groundtruth, iou_list)

    # Calculate AP for orginal images
    for class_ind in range(len(coco_91_classes)):
        for i in range(len(iou_list)):
            origins_ap[class_ind][i] = calculate_AP_for_class(tp[class_ind][i], fp[class_ind][i], gt_instances[class_ind])

    # Calculate TP, FP for exp images
    for folder_i, exp_folder in enumerate(exp_detect_folders):
        tp, fp = calculate_tp_fp_range_from_path(exp_folder, groundtruth, iou_list)

        # Calculate AP for exp images
        for class_ind in range(len(coco_91_classes)):
            for i in range(len(iou_list)):
                exp_APs[folder_i][class_ind][i] = calculate_AP_for_class(tp[class_ind][i], fp[class_ind][i], gt_instances[class_ind])

    write_results(gt_instances, exp_detect_folders, origins_ap, exp_APs)


def write_results(gt_instances, exp_detect_folders, origins_ap, exp_APs):

    filename = f'results.csv'
    f = open(Path(config.exp_dir) / filename, 'w+')

    # Write headers
    results_writer = csv.writer(f, lineterminator='\n')
    headers = ["class", "gt_instances", "original AP@0.5"]
    ap05_headers = []
    ap05095_headers = []
    for folder in exp_detect_folders:
        path_elements = get_last_path_element(folder, 2)
        ap05_headers.append(f"{path_elements[0]} AP@0.5")
        ap05095_headers.append(f"{path_elements[0]} AP@0.5:0.95")
    headers.extend(ap05_headers + ["original AP@0.5:0.95"] + ap05095_headers)
    results_writer.writerow(headers)

    # Write AP results
    ap_results = []
    exp_folders_count = len(exp_detect_folders)
    for class_ind in range(len(coco_91_classes)):
        if gt_instances[class_ind] > 0:
            # AP@0.5 and AP@0.5:0.95 of original
            ap_results.append([coco_91_classes[class_ind+1], gt_instances[class_ind], origins_ap[class_ind][0]])

            # AP@0.5 and AP@0.5:0.95 of exp
            ap05_maps = []
            ap05095_maps = []
            for exp_i in range(exp_folders_count):
                ind = 2 + exp_i
                ap05_maps.append(exp_APs[exp_i][class_ind][0])
                ap05095_maps.append(np.average(exp_APs[exp_i][class_ind]))
            ap_results[-1].extend(ap05_maps + [np.average(origins_ap[class_ind])] + ap05095_maps)

            results_writer.writerow(ap_results[-1])

    # Write mAP results
    map_results = np.array(ap_results)[:, 2:].astype(float)

    # mAP@0.5 and mAP@0.5:0.95 of original
    maps = ['mAP', '', np.average(map_results[:, 0]), np.average(map_results[:, 1])]

    # mAP@0.5 and mAP@0.5:0.95 of exp
    for exp_i in range(exp_folders_count * 2):
        maps.append(np.average(map_results[:, exp_i + 2]))

    results_writer.writerow(maps)

    f.close()


if __name__ == "__main__":
    analyze_config(os.path.abspath('./config.ini'))
    logger = MyLogger.getLog(config.quiet)

    iou_list = np.arange(0.5, 0.95 + 0.05, 0.05)
    gt_data = read_dict_from_json(config.input_annos)["annotations"]
    
    exp_folders = get_exp_folders(config.exp_dir)
    exp_detect_folders = [Path(exp_folder, 'detects') for exp_folder in exp_folders]
    if len(exp_detect_folders) == 0:
        logger.warn(f'No detect folder found for exp_number = {config.exp_number}')

    eval(exp_detect_folders, gt_data, iou_list)
