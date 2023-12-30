import configparser
import sys
from pathlib import Path

import torch

from const.config_const import *
from const.constants import shared_exit_msg
from tests.test_config import test_config
from utility.format_utils import str_to_list_num, str_to_list_str
from utility.mylogger import MyLogger
from utility.path_utils import create_path_if_not_exists, get_last_path_element


class config:
    # general
    input_dir = ""
    image_extensions = []
    batch_size = -1
    plot_count = -1
    groundtruth_json = ""
    device = None

    # analyze frequency
    r_values = []
    force_detect = False

    # experiments
    exp_number = 0
    exp_value_set = []
    exp1_values = []
    exp2_values = []
    force_detect = False

    # model
    model_type = ""
    score_threshold = []

    # output
    output_dir = ""
    analyze_dir = ""
    exp_dir = ""



def analyze_config(config_path):
    logger = MyLogger.getLog()
    logger.debug('Reading configuration. START')

    config_parser = configparser.ConfigParser()
    config_parser.read(config_path, encoding='utf-8-sig')

    # general
    config.input_dir = config_parser[GENERAL_TXT.GENERAL][GENERAL_TXT.input_dir]
    config.image_extensions = str_to_list_str(config_parser[GENERAL_TXT.GENERAL][GENERAL_TXT.image_extensions])
    config.batch_size = int(config_parser[GENERAL_TXT.GENERAL][GENERAL_TXT.batch_size])
    config.plot_count = int(config_parser[GENERAL_TXT.GENERAL][GENERAL_TXT.plot_count])
    config.groundtruth_json = config_parser[GENERAL_TXT.GENERAL][GENERAL_TXT.groundtruth_json]
    # config.groundtruth_data = read_dict_from_json(config.groundtruth_json)
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # analyze frequency
    config.r_values = str_to_list_num(config_parser[ANALYZE_FREQ.ANALYZE_FREQ][ANALYZE_FREQ.r_values])
    config.force_detect = bool(config_parser[ANALYZE_FREQ.ANALYZE_FREQ][ANALYZE_FREQ.force_detect])

    # analyze frequency
    config.exp_number = int(config_parser[EXPERIMENTS.EXPERIMENTS][EXPERIMENTS.exp_number])
    config.exp1_values = str_to_list_num(config_parser[EXPERIMENTS.EXPERIMENTS][EXPERIMENTS.exp1_values])
    config.exp2_values = str_to_list_num(config_parser[EXPERIMENTS.EXPERIMENTS][EXPERIMENTS.exp2_values])
    config.exp_value_set.extend([config.exp1_values, config.exp2_values])
    config.force_detect = bool(config_parser[EXPERIMENTS.EXPERIMENTS][EXPERIMENTS.force_detect])

    # model
    config.model_type = config_parser[MODEL_TXT.MODEL][MODEL_TXT.model_type]
    config.score_threshold = float(config_parser[MODEL_TXT.MODEL][MODEL_TXT.score_threshold])


    # output
    last_folder = get_last_path_element(config.input_dir)
    config.output_dir = Path("output") / last_folder
    create_path_if_not_exists(config.output_dir)

    config.analyze_dir = Path(config.output_dir, "analyze")
    create_path_if_not_exists(config.analyze_dir)

    config.exp_dir = Path(config.output_dir, f"EXP_{config.exp_number}")
    create_path_if_not_exists(config.exp_dir)


    ok = test_config(config, logger)
    if not ok:
        sys.exit(shared_exit_msg)
    logger.debug('Reading configuration. DONE')
    


def get_r_low_dir(r: int):
    low_dir = Path(config.analyze_dir, f"low_{r}")
    create_path_if_not_exists(low_dir)
    return low_dir

def get_r_high_dir(r: int):
    high_dir = Path(config.analyze_dir, f"high_{r}")
    create_path_if_not_exists(high_dir)
    return high_dir
