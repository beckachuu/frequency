import os
import sys

from torch.utils.data import DataLoader

module_path = os.path.abspath(os.getcwd() + "/src")
if module_path not in sys.path:
    sys.path.append(module_path)

from config_parser import analyze_config, config, get_r_high_dir, get_r_low_dir
from dataset import ImageDataset
from freq_analyzer import FrequencyAnalyzer


def main():

    for r in config.r_values:
        dataset = ImageDataset(config.input_dir, config.image_extensions)
        dataloader = DataLoader(dataset=dataset, num_workers=2, batch_size=config.batch_size, shuffle=True)

        low_save_dir = get_r_low_dir(r)
        high_save_dir = get_r_high_dir(r)
        freq_analyzer = FrequencyAnalyzer(low_save_dir, high_save_dir, r)
    
        for i, (images, images_names, images_sizes) in enumerate(dataloader):
            freq_analyzer.generateDataWithDifferentFrequencies(images, images_names, images_sizes)


if __name__ == "__main__":
    analyze_config(os.path.abspath('./config.ini'))
    main()
