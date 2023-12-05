import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from utility.format_utils import resize_auto_interpolation


class FrequencyFilter():
    def __init__(self, low_save_dir, high_save_dir, r=0):
        self.r = r

        self.low_save_dir = low_save_dir
        self.high_save_dir = high_save_dir
        
        self.low_paths = []
        self.high_paths = []

        self.images_freq_low = []
        self.images_freq_high = []

    def distance(self, i, j, imageSize):
        dis = np.sqrt((i - imageSize/2) ** 2 + (j - imageSize/2) ** 2)
        if dis < self.r:
            return 1.0
        else:
            return 0

    def mask_radial(self, img):
        rows, cols = img.shape
        mask = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                mask[i, j] = self.distance(i, j, imageSize=rows)
        return mask
    

    def generateDataWithDifferentFrequencies_3Channel(self, images, images_names, images_sizes) -> (list, list):
        '''
        Image shape must be HWC.
        '''
        self.reset_data()

        mask = self.mask_radial(np.zeros([images.shape[1], images.shape[2]]))
        for i in range(images.shape[0]):
            tmp_low = np.zeros([images.shape[1], images.shape[2], 3])
            tmp_high = np.zeros([images.shape[1], images.shape[2], 3])
            
            for j in range(3):
                fourier_domain = np.fft.fftshift(np.fft.fft2(images[i, :, :, j]))

                fourier_domain_low = fourier_domain * mask
                img_low = np.fft.ifft2(np.fft.ifftshift(fourier_domain_low))
                tmp_low[:,:,j] = np.real(img_low)

                fourier_domain_high = fourier_domain * (1 - mask)
                img_high = np.fft.ifft2(np.fft.ifftshift(fourier_domain_high))
                tmp_high[:,:,j] = np.real(img_high)
            
            height = images_sizes[0][i]
            width = images_sizes[1][i]
            tmp_low = resize_auto_interpolation(tmp_low, height, width)
            tmp_high = resize_auto_interpolation(tmp_high, height, width)
            self.save_result(tmp_low, tmp_high, images_names[i])


    def save_result(self, image_low, image_high, image_name):
        self.images_freq_low.append(image_low)
        self.images_freq_high.append(image_high)

        low_path = Path(self.low_save_dir) / image_name
        high_path = Path(self.high_save_dir) / image_name
        self.low_paths.append(low_path)
        self.high_paths.append(high_path)

        plt.imsave(low_path, image_low.astype(np.uint8))
        plt.imsave(high_path, image_high.astype(np.uint8))


    def get_filepaths(self) -> (list, list):
        return list(self.images_freq_low), list(self.images_freq_high)
    
    def get_filepaths(self) -> (list, list):
        return self.low_paths, self.high_paths
    

    def reset_data(self):
        self.low_paths = []
        self.high_paths = []

        self.images_freq_low = []
        self.images_freq_high = []
