import os
import sys
from importlib import import_module

from torch.utils.data import DataLoader

module_path = os.path.abspath(os.getcwd() + "/src")
if module_path not in sys.path:
    sys.path.append(module_path)

from config_parser import analyze_config, config
from dataset import ImageDataset
from utility.mylogger import MyLogger


def get_experiment(exp_number):
    try:
        experiment = import_module(f"experiments.exp{exp_number}")
        frequencyExp = experiment.FrequencyExp(config.exp_dir, config.exp_value_set[exp_number-1])
        return frequencyExp
    except ModuleNotFoundError:
        sys.exit(f"Experiment {exp_number} does not exist.")


def run_exp(logger):
    dataset = ImageDataset(config.input_dir, config.image_extensions)
    dataloader = DataLoader(dataset=dataset, num_workers=2, batch_size=config.batch_size, shuffle=True)

    frequency_exp = get_experiment(config.exp_number)

    for i, (images, images_names, images_sizes) in enumerate(dataloader):
        logger.info(f"BATCH {i}")
        frequency_exp.run_experiment(images, images_names, images_sizes)



if __name__ == "__main__":
    analyze_config(os.path.abspath('./config.ini'))
    logger = MyLogger.getLog()
    
    run_exp(logger)
