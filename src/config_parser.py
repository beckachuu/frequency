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
    
    train_dir = ""
    train_split = -1
    train_annos = ""
    val_dir = ""
    val_split = -1
    val_annos = ""
    save_labels_dir = ""

    image_extensions = []
    batch_size = -1
    plot_count = -1
    input_annos = ""
    quiet = False
    device = None

    # analyze frequency
    r_values = []

    # experiments
    exp_number = 0
    exp_value_set = []
    force_exp = False
    plot_analyze = False
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

    config.batch_size = int(config_parser[GENERAL_TXT.GENERAL][GENERAL_TXT.batch_size])
    config.input_annos = config_parser[GENERAL_TXT.GENERAL][GENERAL_TXT.input_annos]

    config.train_dir = config_parser[GENERAL_TXT.GENERAL][GENERAL_TXT.train_dir]
    train_split = config_parser[GENERAL_TXT.GENERAL][GENERAL_TXT.train_split]
    if train_split != '':
        config.train_split = float(train_split)
    config.train_annos = config_parser[GENERAL_TXT.GENERAL][GENERAL_TXT.train_annos]

    config.val_dir = config_parser[GENERAL_TXT.GENERAL][GENERAL_TXT.val_dir]
    val_split = config_parser[GENERAL_TXT.GENERAL][GENERAL_TXT.val_split]
    if val_split != '':
        config.val_split = float(val_split)
    config.val_annos = config_parser[GENERAL_TXT.GENERAL][GENERAL_TXT.val_annos]

    config.save_labels_dir = config_parser[GENERAL_TXT.GENERAL][GENERAL_TXT.save_labels_dir]

    config.image_extensions = str_to_list_str(config_parser[GENERAL_TXT.GENERAL][GENERAL_TXT.image_extensions])
    config.plot_count = int(config_parser[GENERAL_TXT.GENERAL][GENERAL_TXT.plot_count])
    config.quiet = bool(config_parser[GENERAL_TXT.GENERAL][GENERAL_TXT.quiet])
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # analyze frequency
    config.r_values = str_to_list_num(config_parser[ANALYZE_FREQ.ANALYZE_FREQ][ANALYZE_FREQ.r_values])

    # experiments
    config.exp_number = int(config_parser[EXPERIMENTS.EXPERIMENTS][EXPERIMENTS.exp_number])
    exp_i = 0
    while True:
        try:
            exp_i += 1
            exp_values = config_parser[EXPERIMENTS.EXPERIMENTS][EXPERIMENTS.exp_values + str(exp_i)]
            try:
                config.exp_value_set.append(str_to_list_num(exp_values))
            except ValueError:
                logger.error(f'In config file: exp_values for EXP{exp_i} is invalid.')
                config.exp_value_set.append('')
        except KeyError:
            break
    config.force_exp = bool(config_parser[EXPERIMENTS.EXPERIMENTS][EXPERIMENTS.force_exp])
    config.plot_analyze = bool(config_parser[EXPERIMENTS.EXPERIMENTS][EXPERIMENTS.plot_analyze])
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
