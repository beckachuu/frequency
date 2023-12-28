import os
import sys

import cv2
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

    if image.shape[0] * image.shape[1] < yolov5_input_size * yolov5_input_size:
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
        results.append(resize_auto_interpolation(image, height, width))
    return results


def preprocess_image_from_url_to_1(img_path, resize_yolo=True):
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

def HWC_to_CHW(image):
    return image.transpose((2, 0, 1))

def BGR_to_RBG(image):
    return image[..., ::-1]


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
    s_list = s.split(',')
    return s_list


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

def str_to_list_num(s: str) -> list:
    '''
    s: string contain multiple elements, separated by comma (may include space)
    '''
    if s.find('.') != -1:
        return str_to_list_float(s)
    return str_to_list_int(s)

