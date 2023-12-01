import glob
import os
import sys

import cv2
import h5py
import numpy as np

module_path = os.path.abspath(os.getcwd() + "/src")
if module_path not in sys.path:
    sys.path.append(module_path)

from const.constants import yolov5_input_size


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

def preprocess_image_from_url_to_1(img_path, resize_yolo=True):
    image, height, width = preprocess_image_from_url_to_255HWC(img_path, resize_yolo)
    image = image.astype(np.float32) / 255
    return image, height, width

def preprocess_image_from_url_to_255HWC(img_path, resize_yolo=True):
    image = cv2.imread(img_path)
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

def prepare_data_aug_DerainDrop(data_path, rain_path, clean_path, patch_size, patches, val_split=0.15):
    # train
    print("Processing training data")

    save_train_target_path = os.path.join(data_path, str(patch_size) + "_train_aug_target.h5")
    save_train_input_path = os.path.join(data_path, str(patch_size) + "_train_aug_input.h5")
    save_val_target_path = os.path.join(data_path, str(patch_size) + "_val_aug_target.h5")
    save_val_input_path = os.path.join(data_path, str(patch_size) + "_val_aug_input.h5")

    train_target_h5f = h5py.File(save_train_target_path, "w")
    train_input_h5f = h5py.File(save_train_input_path, "w")
    val_target_h5f = h5py.File(save_val_target_path, "w")
    val_input_h5f = h5py.File(save_val_input_path, "w")

    get_val_point = int((1 - val_split) / val_split)
    image_count = 0
    val_num = 0
    train_num = image_count - val_num

    clean_paths = glob.glob(clean_path + "/*.png")
    clean_paths += glob.glob(clean_path + "/*.jpg")
    print(f"Number of input clean images: {len(clean_paths)}")
    clean_paths = sorted(clean_paths)

    flipping = True
    for index, clean_path in enumerate(clean_paths):
        filename = clean_path.split("\\")[-1]

        # print(f"train_num: {train_num}, val_num: {val_num}")
        # print(f"Looping at image {index}: {clean_path}")

        clean_img = cv2.imread(clean_path)
        # Handle if clean image has only 2 dimensions (greyscale image)
        if len(clean_img.shape) == 2:
            clean_img = np.repeat(clean_img[..., np.newaxis], 3, -1)
        b, g, r = cv2.split(clean_img)
        clean_img = cv2.merge([r, g, b])

        rain_img = cv2.imread(os.path.join(rain_path, filename))
        b, g, r = cv2.split(rain_img)
        rain_img = cv2.merge([r, g, b])

        H, W, C = clean_img.shape

        size = patch_size
        # max_distance = 0
        for i in range(patches):
            try:
                x1 = np.random.randint(0, W - size)
                y1 = np.random.randint(0, H - size)
            except:
                print("IGNORE THIS (size too small)")
                continue

            x2 = x1 + size
            y2 = y1 + size

            crop_input = rain_img[y1:y2, x1:x2]
            crop_target = clean_img[y1:y2, x1:x2]

            if flipping:
                flipping = False
                crop_input = cv2.flip(crop_input, 1)
                crop_target = cv2.flip(crop_target, 1)
            else:
                flipping = True

            # distance = np.mean(abs(crop_input - crop_target))
            # if distance <= max_distance:
            #     i = max(0, i - 1)
            #     continue
            # else:
            #     max_distance = distance

            # crop_input_flip = input_img_f[y1:y2, x1:x2]
            # crop_target_flip = target_f[y1:y2, x1:x2]

            input_img_normal = np.float32(crop_input / 255.)
            target_img_normal = np.float32(crop_target / 255.)
            # input_img_flip = np.float32(normalize(crop_input_flip))
            # target_img_flip = np.float32(normalize(crop_target_flip))

            input_data_1 = input_img_normal.transpose(2, 0, 1).copy()
            target_data_1 = target_img_normal.transpose(2, 0, 1).copy()
            if index % get_val_point == 0:
                val_input_h5f.create_dataset(str(val_num), data=input_data_1)
                val_target_h5f.create_dataset(str(val_num), data=target_data_1)
                val_num = val_num + 1
            else:
                train_input_h5f.create_dataset(str(train_num), data=input_data_1)
                train_target_h5f.create_dataset(str(train_num), data=target_data_1)
                train_num = train_num + 1

            # plt.imshow(np.transpose(input_data_1, (1, 2, 0)))
            # plt.show()
            # plt.imshow(np.transpose(input_data_flip, (1, 2, 0)))
            # plt.show()

            # if input_data_1.shape[1] <=1 or input_data_2.shape[1] <=1 or target_data_1.shape[1] <=1 or target_data_2.shape[1] <=1:
            #     print('wrong', input_data_1.shape, input_data_2.shape, target_data_1.shape, target_data_2.shape)

    train_target_h5f.close()
    train_input_h5f.close()
    val_target_h5f.close()
    val_input_h5f.close()

    print(f"FINISH PREPROCESS DATA: train_num = {train_num}, val_num = {val_num}")


def str_to_list_str(s: str) -> list:
    '''
    s: string contain multiple elements, separated by comma (may include space)
    '''
    s = s.strip()
    s_list = s.split(',')
    return s_list


def str_to_list_int(s: str) -> list:
    '''
    s: string contain multiple elements, separated by comma (may include space)
    '''
    s = s.strip()
    s_list = s.split(',')
    int_list = [int(s) for s in s_list]
    return int_list


def str_to_list_float(s: str) -> list:
    '''
    s: string contain multiple elements, separated by comma (may include space)
    '''
    s = s.strip()
    s_list = s.split(',')
    int_list = [float(s) for s in s_list]
    return int_list


