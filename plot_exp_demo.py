import os
import sys
from pathlib import Path

module_path = os.path.abspath(os.getcwd() + "/src")
if module_path not in sys.path:
    sys.path.append(module_path)

from config_parser import analyze_config, config
from utility.format_utils import preprocess_image_from_url_to_255HWC
from utility.map_util import get_detect_bbox, get_gt_bbox
from utility.mylogger import MyLogger
from utility.path_utils import (get_exp_folders, get_filepaths_list,
                                get_last_path_element, read_dict_from_json)
from utility.plot_util import plot_images_with_bbox


def plot(groundtruth, exp_folders, logger):
    origins_paths = get_filepaths_list(Path(config.input_dir), config.image_extensions)
    plot_count = 0

    for i, file_path in enumerate(origins_paths):
        
        if plot_count >= config.plot_count and config.plot_count >= 0:
            break

        image_name = get_last_path_element(file_path)
        image_id = image_name.split('.')[0]
        gt_id = int(image_id)

        logger.info(f'Plotting demo for: {image_name}')

        images = []
        bboxes = []
        titles = ['ground truth', 'detect']

        gt_bbox = get_gt_bbox(groundtruth, gt_id)

        origin_image, h, w = preprocess_image_from_url_to_255HWC(file_path, resize_yolo=False)
        detect_file = Path(config.output_dir, 'detects', f'{str(image_id)}.txt')
        detect_data = get_detect_bbox(detect_file)

        images.extend([origin_image, origin_image])
        bboxes.extend([gt_bbox, detect_data])

        no_correspond = False
        for exp_folder in exp_folders:
            exp_img_path = Path(exp_folder, image_name)

            if not os.path.isfile(exp_img_path):
                logger.error(f'No correspoding image in {exp_folder}. Skipping this image.')
                no_correspond = True
                break

            analyze_image, h, w = preprocess_image_from_url_to_255HWC(exp_img_path, resize_yolo=False)
            detect_file = Path(exp_folder, 'detects', f'{str(image_id)}.txt')
            detect_data = get_detect_bbox(detect_file)

            images.extend([analyze_image])
            bboxes.extend([detect_data])

            path_elements = get_last_path_element(exp_folder, 2)
            titles.extend([f"{path_elements[0]}/{path_elements[1]}"])

        if not no_correspond:
            plot_images_with_bbox(images, bboxes, titles, fig_save_path=Path(config.exp_dir, image_name))
            plot_count += 1



if __name__ == "__main__":
    analyze_config(os.path.abspath('./config.ini'))
    logger = MyLogger.getLog()

    gt_data = read_dict_from_json(config.groundtruth_json)["annotations"]
    exp_folders = get_exp_folders(config.exp_dir)

    plot(gt_data, exp_folders, logger)

