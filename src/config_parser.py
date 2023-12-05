import configparser
import os
import sys
from pathlib import Path

import torch

from const.config_const import FREQUENCY_TXT, GENERAL_TXT, MODEL_TXT
from const.constants import shared_exit_msg
from tests.test_config import test_config
from utility.format_utils import str_to_list_int, str_to_list_str
from utility.mylogger import MyLogger
from utility.path_utils import get_last_path_element


class config:
    # general
    input_dir = ""
    image_extensions = []
    batch_size = -1
    demo_count = -1
    groundtruth_json = ""
    device = None

    # frequency
    r_values = []

    # model
    model_type = ""
    score_threshold = []

    output_dir = ""



def analyze_config(config_path):
    logger = MyLogger.getLog()
    logger.debug('Reading configuration. START')

    config_parser = configparser.ConfigParser()
    config_parser.read(config_path, encoding='utf-8-sig')

    # general
    config.input_dir = config_parser[GENERAL_TXT.GENERAL][GENERAL_TXT.input_dir]

    config.image_extensions = str_to_list_str(config_parser[GENERAL_TXT.GENERAL][GENERAL_TXT.image_extensions])

    config.batch_size = int(config_parser[GENERAL_TXT.GENERAL][GENERAL_TXT.batch_size])

    config.demo_count = int(config_parser[GENERAL_TXT.GENERAL][GENERAL_TXT.demo_count])

    config.groundtruth_json = config_parser[GENERAL_TXT.GENERAL][GENERAL_TXT.groundtruth_json]
    # config.groundtruth_data = read_dict_from_json(config.groundtruth_json)

    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # frequency
    config.r_values = str_to_list_int(config_parser[FREQUENCY_TXT.FREQUENCY][FREQUENCY_TXT.r_values])

    # model
    config.model_type = config_parser[MODEL_TXT.MODEL][MODEL_TXT.model_type]
    config.score_threshold = float(config_parser[MODEL_TXT.MODEL][MODEL_TXT.score_threshold])

    # output
    last_folder = get_last_path_element(config.input_dir)
    config.output_dir = Path("output") / last_folder


    ok = test_config(config, logger)
    if not ok:
        sys.exit(shared_exit_msg)
    logger.debug('Reading configuration. DONE')
    


def get_r_low_dir(r: int):
    low_dir = Path(config.output_dir, f"low_{r}")
    if not os.path.exists(low_dir):
        os.makedirs(low_dir)
    return low_dir

def get_r_high_dir(r: int):
    high_dir = Path(config.output_dir, f"high_{r}")
    if not os.path.exists(high_dir):
        os.makedirs(high_dir)
    return high_dir
