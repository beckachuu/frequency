import os

from utility.path_utils import get_project_root

TOOL_NAME = 'FreqFilter'


# paths
ROOT_PATH = get_project_root()

OUTPUT_DETECT_FOLDER_PATH = os.path.join(ROOT_PATH, 'output', 'detects')
DETECT_TEMPLATE_FILE_PATH = os.path.join(OUTPUT_DETECT_FOLDER_PATH, 'result_template.txt')

OUTPUT_ATTACK_FOLDER_PATH = os.path.join(ROOT_PATH, 'output', 'attacks')
ATTACK_TEMPLATE_FILE_PATH = os.path.join(OUTPUT_ATTACK_FOLDER_PATH, 'result_template.txt')

OUTPUT_DEFENSE_FOLDER_PATH = os.path.join(ROOT_PATH, 'output', 'mitigates')
MITIGATE_TEMPLATE_FILE_PATH = os.path.join(OUTPUT_DEFENSE_FOLDER_PATH, 'result_template.txt')

CHECKPOINTS_FOLDER_PATH = os.path.join(ROOT_PATH, 'checkpoints')

IMAGES_FOLDER = 'images'
NPY_FOLDER = 'npy'
DEMO_FOLDER = 'demo'
ADV_FOLDER = 'adv'
DEPTH_FOLDER = 'depth'
DEFOG_FOLDER = 'defog'
ORIGIN_FOLDER = 'origin'
MITIGATED_FOLDER = 'mitigated_adv'



# default configuration
yolov5_input_size = 640
channels = 3



# default error message
shared_exit_msg = 'Please check out configuration!'
shared_incorrect_para_msg = '{param} is not correct.'

