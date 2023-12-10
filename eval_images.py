import csv
import os
import sys
from pathlib import Path

from mapcalc import calculate_map, calculate_map_range

module_path = os.path.abspath(os.getcwd() + "/src")
if module_path not in sys.path:
    sys.path.append(module_path)

from config_parser import analyze_config, config, get_r_low_dir
from utility.map_util import get_detect_list, load_detect_data, load_gt_data
from utility.path_utils import (get_filepaths_list, get_last_path_element,
                                read_dict_from_json)


def eval(groundtruth):

    f = open(Path(config.output_dir) / f'results_images_{config.r_values}.csv', 'w+')
    results_writer = csv.writer(f, lineterminator='\n')
    headers = ["ID", "Original mAP@0.5", "Original mAP@0.5:0.95"]
    for r in config.r_values:
        headers.extend([f"Low_{r} mAP@0.5", f"Low_{r} mAP@0.5:0.95"])
    results_writer.writerow(headers)

    origins_id = []

    origins_map05, origins_map095 = [], []
    lows_map05 = [[] for _ in range(len(config.r_values))]
    lows_map095 = [[] for _ in range(len(config.r_values))]


    origins_paths = get_filepaths_list(Path(config.output_dir) / 'detects', ['txt'])
    for i, file_path in enumerate(origins_paths):
        image_id = int(get_last_path_element(file_path).split('.')[0])
        origins_id.append(image_id)

        gt_data = load_gt_data(groundtruth, image_id)
        detect_data = load_detect_data(get_detect_list(file_path))
        map_value = calculate_map(gt_data, detect_data, 0.5)
        origins_map05.append(map_value)
        map_value = calculate_map_range(gt_data, detect_data, 0.5, 0.95, 0.05)
        origins_map095.append(map_value)
    
    for r_i, radius in enumerate(config.r_values):
        low_path = get_r_low_dir(radius)
        low_paths = get_filepaths_list(Path(low_path) / 'detects', ['txt'])

        for i, file_path in enumerate(low_paths):
            image_id = int(get_last_path_element(file_path).split('.')[0])

            gt_data = load_gt_data(groundtruth, image_id)
            detect_data = load_detect_data(get_detect_list(file_path))
            map_value = calculate_map(gt_data, detect_data, 0.5)
            lows_map05[r_i].append(map_value)
            map_value = calculate_map_range(gt_data, detect_data, 0.5, 0.95, 0.05)
            lows_map095[r_i].append(map_value)

    for i in range(len(origins_id)):
        results = [origins_id[i], origins_map05[i], origins_map095[i]]
        for r_i in range(len(config.r_values)):
            results.extend([lows_map05[r_i][i], lows_map095[r_i][i]])
        results_writer.writerow(results)

    f.close()


if __name__ == "__main__":
    analyze_config(os.path.abspath('./config.ini'))

    gt_data = read_dict_from_json(config.groundtruth_json)["annotations"]
    eval(gt_data)
