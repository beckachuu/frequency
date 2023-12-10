import csv
import os
import sys
from pathlib import Path

import numpy as np

module_path = os.path.abspath(os.getcwd() + "/src")
if module_path not in sys.path:
    sys.path.append(module_path)

from config_parser import analyze_config, config, get_r_low_dir
from const.constants import coco_91_classes
from utility.map_util import (calculate_AP_for_class,
                              calculate_tp_fp_range_for_class, get_detect_list,
                              load_detect_data, load_gt_data)
from utility.path_utils import (get_filepaths_list, get_last_path_element,
                                read_dict_from_json)


def eval(groundtruth, iou_list):

    f = open(Path(config.output_dir) / f'results_{config.r_values}.csv', 'w+')
    results_writer = csv.writer(f, lineterminator='\n')
    headers = ["Class", "Instances", "Original AP@0.5", "Original AP@0.5:0.95"]
    for r in config.r_values:
        headers.extend([f"Low_{r} AP@0.5", f"Low_{r} AP@0.5:0.95"])
    results_writer.writerow(headers)
    
    instances = [0] * len(coco_91_classes)
    origins_ap = [[0 for _ in range(len(iou_list))] for _ in range(len(coco_91_classes))]
    lows_ap = [[[0 for _ in range(len(iou_list))] for _ in range(len(coco_91_classes))] for _ in range(len(config.r_values))]
    tp = [[[] for _ in range(len(iou_list))] for _ in range(len(coco_91_classes))]
    fp = [[[] for _ in range(len(iou_list))] for _ in range(len(coco_91_classes))]

    # Calculate true positive, false positive for orginal images
    origins_paths = get_filepaths_list(Path(config.output_dir) / 'detects', ['txt'])
    for file_path in origins_paths:
        image_id = int(get_last_path_element(file_path).split('.')[0])
        gt_data = load_gt_data(groundtruth, image_id)
        detect_data = load_detect_data(get_detect_list(file_path))

        # accumulate_tp_fp for each class
        for class_ind in gt_data['labels']:
            tp_range, fp_range = calculate_tp_fp_range_for_class(gt_data, detect_data, class_ind)
            for i in range(len(iou_list)):
                tp[class_ind-1][i] += tp_range[i]
                fp[class_ind-1][i] += fp_range[i]
            
            instances[class_ind-1] += gt_data['labels'].count(class_ind)

    # Calculate AP for orginal images
    for class_ind in range(len(coco_91_classes)):
        for i in range(len(iou_list)):
            origins_ap[class_ind][i] = calculate_AP_for_class(tp[class_ind][i], fp[class_ind][i], instances[class_ind])


    # Calculate true positive, false positive for low frequency images
    for r_i, radius in enumerate(config.r_values):
        tp = [[[] for _ in range(len(iou_list))] for _ in range(len(coco_91_classes))]
        fp = [[[] for _ in range(len(iou_list))] for _ in range(len(coco_91_classes))]

        low_path = get_r_low_dir(radius)
        low_paths = get_filepaths_list(Path(low_path) / 'detects', ['txt'])
        for file_path in low_paths:
            image_id = int(get_last_path_element(file_path).split('.')[0])
            gt_data = load_gt_data(groundtruth, image_id)
            detect_data = load_detect_data(get_detect_list(file_path))

            # accumulate_tp_fp for each class
            for class_ind in gt_data['labels']:
                tp_range, fp_range = calculate_tp_fp_range_for_class(gt_data, detect_data, class_ind)
                for i in range(len(iou_list)):
                    tp[class_ind-1][i] += tp_range[i]
                    fp[class_ind-1][i] += fp_range[i]

        # Calculate AP for low frequency images
        for class_ind in range(len(coco_91_classes)):
            for i in range(len(iou_list)):
                lows_ap[r_i][class_ind][i] = calculate_AP_for_class(tp[class_ind][i], fp[class_ind][i], instances[class_ind])

    # Write AP results
    all_map = []
    for class_ind in range(len(coco_91_classes)):
        if instances[class_ind] > 0:
            all_map.extend([[0] * (2 + 2 * len(config.r_values))])
            all_map[-1][0] += origins_ap[class_ind][0]
            all_map[-1][1] += np.average(origins_ap[class_ind])
            results = [coco_91_classes[class_ind+1], instances[class_ind], all_map[-1][0], all_map[-1][1]]
            for r_i in range(len(config.r_values)):
                ind = 2 + r_i*2
                all_map[-1][ind] += lows_ap[r_i][class_ind][0]
                all_map[-1][ind+1] += np.average(lows_ap[r_i][class_ind])
                results.extend([all_map[-1][ind], all_map[-1][ind+1]])
            results_writer.writerow(results)

    # Write mAP results
    all_map = np.array(all_map)
    results = ['', 'mAP', np.average(all_map[:, 0]), np.average(all_map[:, 1])]
    for r_i in range(len(config.r_values)):
        ind = 2 + r_i*2
        results.extend([np.average(all_map[:, ind]), np.average(all_map[:, ind+1])])
    results_writer.writerow(results)

    f.close()


if __name__ == "__main__":
    analyze_config(os.path.abspath('./config.ini'))

    iou_list = np.arange(0.5, 0.95 + 0.05, 0.05)
    gt_data = read_dict_from_json(config.groundtruth_json)["annotations"]
    eval(gt_data, iou_list)
