import glob
import json
import os
import sys
from datetime import datetime

from natsort import natsorted, ns


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_timestamp(timestamp_format="%d%m%y-%H%M%S"):
    return datetime.now().strftime(timestamp_format)


def create_path_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


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
    filepaths.sort()
    return filepaths


def count_filepaths(folder_path: str, extensions: list = None) -> int:
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist.")
        return 0

    if not extensions:
        return len([name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))])
    else:
        filepaths = []
        for ext in extensions:
            filepaths += glob.glob(os.path.join(folder_path, f"*.{ext}"))
        return len(filepaths)
    
def check_files_exist(file_paths: list) -> bool:
    for file_path in file_paths:
        if not os.path.isfile(file_path):
            return False
    return True


def get_last_path_element(path: str, n = 1):
    elements = os.path.normpath(path).split(os.sep)
    if n == 1:
        return elements[-1]
    return elements[-n:]


def get_child_folders(path, level=1, nat_sort=True):
    if level == 1:
        folders = [os.path.join(path, name) for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    else:
        folders = []
        for name in os.listdir(path):
            if os.path.isdir(os.path.join(path, name)):
                folders.extend(get_child_folders(os.path.join(path, name), level - 1, False))

    if nat_sort:
        folders = natsorted(folders, alg=ns.IGNORECASE)
    
    return folders

def get_exp_folders(exp_dir, exclude=["checkpoints"]) -> list:
    '''
        Get result folders of an exp directory
        Return 1st level child folders (or parent folder if no child found)
    '''
    exp_folders = get_child_folders(exp_dir, 1)
    exp_folders = list(filter(lambda folder: not any(word in folder for word in exclude), exp_folders))
    if len(exp_folders) == 0:
        exp_folders = [exp_dir]

    return exp_folders

