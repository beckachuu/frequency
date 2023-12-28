import os
import sys
from importlib import import_module

from torch.utils.data import DataLoader

module_path = os.path.abspath(os.getcwd() + "/src")
if module_path not in sys.path:
    sys.path.append(module_path)

from config_parser import analyze_config, config
from dataset import ImageDataset


def get_experiment():
    try:
        experiment = import_module(f"experiments.exp{config.exp_number}")
        frequencyExp = experiment.FrequencyExp(config.exp_dir, config.exp_values)
        return frequencyExp
    except ModuleNotFoundError:
        sys.exit(f"Experiment {config.exp_number} does not exist.")


def run_exp():
    dataset = ImageDataset(config.input_dir, config.image_extensions)
    dataloader = DataLoader(dataset=dataset, num_workers=2, batch_size=config.batch_size, shuffle=True)

    frequency_exp = get_experiment()

    for i, (images, images_names, images_sizes) in enumerate(dataloader):
        frequency_exp.run_experiment(images, images_names, images_sizes)



if __name__ == "__main__":
    analyze_config(os.path.abspath('./config.ini'))
    
    run_exp()
