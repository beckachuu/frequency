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
    analyzes_ap = [[[0 for _ in range(len(iou_list))] for _ in range(len(coco_91_classes))] for _ in range(len(exp_detect_folders))]
    
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

    # Calculate true positive, false positive for analyze images
    for path_ind, analyze_path in enumerate(exp_detect_folders):
        # tp = [[[] for _ in range(len(iou_list))] for _ in range(len(coco_91_classes))]
        # fp = [[[] for _ in range(len(iou_list))] for _ in range(len(coco_91_classes))]

        # analyze_files_paths = get_filepaths_list(analyze_path, ['txt'])
        # for file_path in analyze_files_paths:
        #     image_id = int(get_last_path_element(file_path).split('.')[0])
        #     gt_data = get_gt_data(groundtruth, image_id)
        #     detect_data = load_detect_data(get_detect_list(file_path))

        #     # accumulate_tp_fp for each class
        #     for class_ind in gt_data['labels']:
        #         tp_range, fp_range = calculate_tp_fp_range_for_class(gt_data, detect_data, class_ind)
        #         for i in range(len(iou_list)):
        #             tp[class_ind-1][i] += tp_range[i]
        #             fp[class_ind-1][i] += fp_range[i]

        tp, fp = calculate_tp_fp_range_from_path(analyze_path, groundtruth, iou_list)

        # Calculate AP for low frequency images
        for class_ind in range(len(coco_91_classes)):
            for i in range(len(iou_list)):
                analyzes_ap[path_ind][class_ind][i] = calculate_AP_for_class(tp[class_ind][i], fp[class_ind][i], gt_instances[class_ind])

    write_results(gt_instances, exp_detect_folders, origins_ap, analyzes_ap)


def write_results(gt_instances, exp_detect_folders, origins_ap, analyzes_ap):

    filename = f'results.csv'
    f = open(Path(config.exp_dir) / filename, 'w+')

    results_writer = csv.writer(f, lineterminator='\n')
    headers = ["class", "gt_instances", "original AP@0.5", "original AP@0.5:0.95"]
    for folder in exp_detect_folders:
        path_elements = get_last_path_element(folder, 2)
        headers.extend([f"{path_elements[0]} AP@0.5", f"{path_elements[0]} AP@0.5:0.95"])
    results_writer.writerow(headers)

    # Write AP results
    all_map = []
    for class_ind in range(len(coco_91_classes)):
        if gt_instances[class_ind] > 0:
            all_map.extend([[0] * (2 + 2 * len(exp_detect_folders))])
            all_map[-1][0] += origins_ap[class_ind][0]
            all_map[-1][1] += np.average(origins_ap[class_ind])
            results = [coco_91_classes[class_ind+1], gt_instances[class_ind], all_map[-1][0], all_map[-1][1]]
            for r_i in range(len(exp_detect_folders)):
                ind = 2 + r_i*2
                all_map[-1][ind] += analyzes_ap[r_i][class_ind][0]
                all_map[-1][ind+1] += np.average(analyzes_ap[r_i][class_ind])
                results.extend([all_map[-1][ind], all_map[-1][ind+1]])
            results_writer.writerow(results)

    # Write mAP results
    all_map = np.array(all_map)
    results = ['', 'mAP', np.average(all_map[:, 0]), np.average(all_map[:, 1])]
    for r_i in range(len(exp_detect_folders)):
        ind = 2 + r_i*2
        results.extend([np.average(all_map[:, ind]), np.average(all_map[:, ind+1])])
    results_writer.writerow(results)

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
