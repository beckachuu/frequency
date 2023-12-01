import glob
import os

import cv2
import h5py
import numpy as np
import torch.utils.data as udata

from utility.format_utils import preprocess_image_from_url_to_255HWC
from utility.path_utils import get_filepaths_list, get_last_path_element


class ImageDataset(udata.Dataset):
    def __init__(self, input_dir, image_extensions):
        super(ImageDataset, self).__init__()

        self.input_dir = input_dir
        self.filepaths = get_filepaths_list(input_dir, image_extensions)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, index: int):
        img, h, w = preprocess_image_from_url_to_255HWC(self.filepaths[index])
        image_name = get_last_path_element(self.filepaths[index])

        return img, image_name, (h, w)


class PathDataset(udata.Dataset):
    def __init__(self, input_dir, image_extensions):
        super(PathDataset, self).__init__()

        self.input_dir = input_dir
        self.filepaths = get_filepaths_list(input_dir, image_extensions)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, index: int):
        return self.filepaths[index]
    

def prepare_h5py_data(data_path, rain_path, path, val_split=0.15):
    print("Processing data")

    save_path = os.path.join(data_path + "data.h5")
    h5f = h5py.File(save_path, "w")

    paths = glob.glob(path + "/*.png")
    paths += glob.glob(path + "/*.jpg")
    print(f"Number of input images: {len(paths)}")
    paths = sorted(paths)

    flipping = True
    for index, path in enumerate(paths):
        img = cv2.imread(path)
        # Handle if clean image has only 2 dimensions (greyscale image)
        if len(img.shape) == 2:
            img = np.repeat(img[..., np.newaxis], 3, -1)
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])


        if flipping:
            flipping = False
            img = cv2.flip(img, 1)
        else:
            flipping = True

        # distance = np.mean(abs(crop_input - crop_target))
        # if distance <= max_distance:
        #     i = max(0, i - 1)
        #     continue
        # else:
        #     max_distance = distance

        img = np.float32(img) / 255.
        img = img.transpose(2, 0, 1).copy()

        h5f.create_dataset(str(index), data=img)


    h5f.close()

    print(f"FINISH PREPROCESS DATA: {len(paths)}")

