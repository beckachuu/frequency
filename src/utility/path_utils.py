import glob
import json
import os
import sys
from datetime import datetime

import torch
import torchvision


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_timestamp(timestamp_format="%d%m%y-%H%M%S"):
    return datetime.now().strftime(timestamp_format)


def check_path_exists(path):
    return os.path.exists(path)


def mkdir(path):
    if not check_path_exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def clear_dir(dir_path):
    if not any(os.scandir(dir_path)):
        return
    for f in os.listdir(dir_path):
        os.remove(os.path.join(dir_path, f))


def check_file_extension(file_path: str, extension_type: str):
    return file_path.endswith(extension_type)


def get_file_name(file_path: str):
    full_file_name = os.path.basename(file_path)
    return os.path.splitext(full_file_name)[0]


def get_parent_dir(sub_path: str):
    return os.path.dirname(sub_path)


def write_to_file(content, path: str, mode='w'):
    file_writer = open(file=path, mode=mode)
    file_writer.write(content)
    file_writer.close()


def img_is_color(img):
    if len(img.shape) == 3:
        # Check the color channels to see if they're all the same.
        c1, c2, c3 = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        if (c1 == c2).all() and (c2 == c3).all():
            return True

    return False


def save_image_to_folder(folder_path, image, prefix_file_name='', extension='png'):
    if not check_path_exists(folder_path):
        mkdir(folder_path)

    # images = np.clip(images, a_min=0, a_max=1)

    file_name = prefix_file_name + '.' + extension
    file_path = os.path.join(folder_path, file_name)
    torchvision.utils.save_image(torch.Tensor(image.transpose((2, 0, 1))), file_path)


def read_txt_from_file(path_to_file):
    with open(path_to_file) as f:
        contents = f.read()
    return contents

def read_dict_from_json(json_dir):
    try:
        if json_dir != "":
            with open(json_dir) as f:
                json_dict = json.load(f)
            return json_dict
    except FileNotFoundError:
        sys.exit(f'\nCan\'t find groundtruth JSON file in the path you provided."')
    except json.JSONDecodeError as _:
        # import traceback
        # traceback.print_exc()
        sys.exit('\nYour groundtruth file has some syntax errors.')


def get_filepaths_list(folder_path: str, extensions: list) -> list:
    filepaths = []
    for ext in extensions:
        filepaths += glob.glob(os.path.join(folder_path, f"*.{ext}"))
    return filepaths

def get_last_path_element(path: str) -> str:
    elements = os.path.normpath(path).split(os.sep)
    return elements[-1]
