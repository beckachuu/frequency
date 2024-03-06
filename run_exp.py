import os
import sys
from importlib import import_module

module_path = os.path.abspath(os.getcwd() + "/src")
if module_path not in sys.path:
    sys.path.append(module_path)

from config_parser import analyze_config, config
from utility.mylogger import MyLogger


def get_experiment(exp_number):
    try:
        experiment = import_module(f"experiments.exp{exp_number}")
        frequencyExp = experiment.FrequencyExp(logger, config.input_dir, config.image_extensions, config.batch_size,
                                               config.exp_dir, config.exp_value_set[exp_number-1], 
                                               config.force_exp, config.plot_analyze)
        return frequencyExp
    except ModuleNotFoundError:
        sys.exit(f"Experiment {exp_number} does not exist.")


def run_exp(logger):
    frequency_exp = get_experiment(config.exp_number)

    logger.info(f'[EXP {config.exp_number}]: force_exp = {config.force_exp}, plot_analyze = {config.plot_analyze}')

    if config.exp_number == 4:
        frequency_exp.run_experiment(config.train_dir, config.train_split, config.train_annos,
                                     config.val_dir, config.val_split, config.val_annos,
                                     config.save_labels_dir, config.model_type, config.model_weights)
    else:
        frequency_exp.run_experiment()



if __name__ == "__main__":
    analyze_config(os.path.abspath('./config.ini'))
    logger = MyLogger.getLog(config.quiet)
    
    run_exp(logger)
