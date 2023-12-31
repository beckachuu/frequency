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
RGB_CHANNELS_NAMES = ['red', 'green', 'blue']


# default error message
shared_exit_msg = 'Please check out configuration!'
shared_incorrect_para_msg = '{param} is invalid.'


coco_91_classes= {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorbike',
    5: 'aeroplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'trafficlight',
    11: 'firehydrant',
    12: 'streetsign',
    13: 'stopsign',
    14: 'parkingmeter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    26: 'hat',
    27: 'backpack',
    28: 'umbrella',
    29: 'shoe',
    30: 'eyeglasses',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sportsball',
    38: 'kite',
    39: 'baseballbat',
    40: 'baseballglove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennisracket',
    44: 'bottle',
    45: 'plate',
    46: 'wineglass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hotdog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'sofa',
    64: 'pottedplant',
    65: 'bed',
    66: 'mirror',
    67: 'diningtable',
    68: 'window',
    69: 'desk',
    70: 'toilet',
    71: 'door',
    72: 'tvmonitor',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cellphone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    83: 'blender',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddybear',
    89: 'hairdrier',
    90: 'toothbrush',
    91: 'hairbrush'
}

epsilon = 1e-7  # small constant

PLOT_FONT_SCALE = 40
