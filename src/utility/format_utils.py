import os
import sys

import cv2
import torch

try:
    import cupy as np
except ImportError:
    import numpy as np

module_path = os.path.abspath(os.getcwd() + "/src")
if module_path not in sys.path:
    sys.path.append(module_path)

from const.constants import epsilon, yolov5_input_size


def resize_auto_interpolation(image: np.ndarray, height=yolov5_input_size, width=yolov5_input_size):
    '''
    @image: must be in the shape of [height, width, channel]
    '''
    height = int(height)
    width = int(width)

    image = to_numpy(image)

    if image.shape[0] * image.shape[1] < height * width:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    else:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image

def resize_auto_interpolation_same(images, height=yolov5_input_size, width=yolov5_input_size):
    '''
    @images: all must be in the shape of [height, width, channel]
    All images will be resized to same size.
    '''
    results = []
    for image in images:
        results.append(resize_auto_interpolation(to_numpy(image), height, width))
    return results

def crop_center(img, crop_h, crop_w):
    y, x = img.shape[:2]
    startx = x//2 - crop_w//2
    starty = y//2 - crop_h//2
    if img.ndim == 3:
        return img[starty:starty+crop_h, startx:startx+crop_w, :]
    else:
        return img[starty:starty+crop_h, startx:startx+crop_w]


def preprocess_image_from_url_to_1HWC(img_path, resize_yolo=True):
    image, height, width = preprocess_image_from_url_to_255HWC(img_path, resize_yolo)
    image = image.astype(np.float32) / 255
    return image, height, width

def preprocess_image_from_url_to_255HWC(img_path, resize_yolo=True):
    image = cv2.imread(str(img_path))
    if len(image.shape) == 2:
        image = np.repeat(image[...,np.newaxis], 3, -1)
    height = image.shape[0]
    width = image.shape[1]
    image = image[..., ::-1] # flips the channel dimension of the image -> BGR to RBG
    if resize_yolo:
        image = resize_auto_interpolation(image)
    return image, height, width

def CHW_to_HWC(image: np.ndarray) -> np.ndarray:
    return image.transpose((1, 2, 0))

def HWC_to_CHW(image: np.ndarray) -> np.ndarray:
    return image.transpose((2, 0, 1))

def BGR_to_RBG(image):
    return image[..., ::-1]

def preprocess_image_from_url_to_torch_input(img_path, resize_yolo=True):
    image, height, width = preprocess_image_from_url_to_1HWC(img_path, resize_yolo)
    image = HWC_to_CHW(image)
    return torch.from_numpy(image), height, width


def complex_to_polar_real(complex_data):
    '''
     Convert the complex image to magnitude and phase
    '''
    magnitude = np.abs(complex_data)

    phase = np.angle(complex_data)
    return magnitude, phase

def polar_real_to_complex(magnitude, phase):
    complex_data = magnitude * np.exp(1j * phase)
    return complex_data

def spatial_real_to_complex(real, imagine):
    complex_data = real + 1j * imagine
    return complex_data


def log_normalize(img: np.ndarray):
    return np.log(img + epsilon)


def align_to_four(img):
    #align to four
    a_row = int(img.shape[0]/4)*4
    a_col = int(img.shape[1]/4)*4

    img = img[0:a_row, 0:a_col]
    return img


def str_to_list_str(s: str) -> list:
    '''
    s: string contain multiple elements, separated by comma (may include space)
    '''
    s = s.replace(' ', '')
    return s.split(',')


def str_to_list_int(s: str) -> list:
    '''
    s: string contain multiple elements, separated by comma (may include space)
    '''
    s = s.replace(' ', '')
    s_list = s.split(',')
    int_list = [int(s) for s in s_list]
    return int_list

def str_to_list_float(s: str) -> list:
    '''
    s: string contain multiple elements, separated by comma (may include space)
    '''
    s = s.replace(' ', '')
    s_list = s.split(',')
    int_list = [float(s) for s in s_list]
    return int_list

def str_to_list_num(s: str):
    '''
    s: string contain multiple elements, separated by comma (may include space)
    Each list is separated by a slash ('/').
    '''
    lists_str = s.split('/')

    if len(lists_str) == 1:
        if lists_str[0].find('.') != -1:
            return str_to_list_float(lists_str[0])
        else:
            return str_to_list_int(lists_str[0])
    else:
        lists_num = []
        for list_str in lists_str:
            if list_str.find('.') != -1:
                lists_num.append(str_to_list_float(list_str))
            else:
                lists_num.append(str_to_list_int(list_str))

        return lists_num


def to_numpy(data):
    if 'numpy' not in str(type(data)):
        data = np.asnumpy(data)
    return data

def to_cupy(data):
    if 'cupy' not in str(type(data)):
        data = np.asarray(data)
    return data

