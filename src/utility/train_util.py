import glob
import os
import re


def find_last_checkpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, '*epoch*.pt'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*epoch(.*).pt.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch

