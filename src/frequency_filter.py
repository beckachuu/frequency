from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from utility.format_utils import (complex_to_polar_real, log_normalize,
                                  resize_auto_interpolation,
                                  resize_auto_interpolation_same)
from utility.mylogger import MyLogger
from utility.path_utils import create_path_if_not_exists
from utility.plot_util import plot_images


class FrequencyFilter():
    def __init__(self, low_save_dir: str, high_save_dir: str, r: int=0):
        self.r = r

        self.low_save_dir = low_save_dir
        self.high_save_dir = high_save_dir

        # Experiments paths
        self.low_analyze_path = Path(low_save_dir) / 'analyze'
        self.high_analyze_path = Path(high_save_dir) / 'analyze'
        create_path_if_not_exists(self.low_analyze_path)
        create_path_if_not_exists(self.high_analyze_path)

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
    

    def generateDataWithDifferentFrequencies(self, images, images_names, images_sizes) -> (list, list):
        '''
        Image shape must be HWC.
        '''
        logger = MyLogger.getLog()

        mask = self.mask_radial(np.zeros([images.shape[1], images.shape[2]]))
        for i in range(images.shape[0]):
            logger.info(f"Experimenting on image: {images_names[i]}")

            image_low = np.zeros([images.shape[1], images.shape[2], 3])
            image_high = np.zeros([images.shape[1], images.shape[2], 3])
            height = images_sizes[0][i]
            width = images_sizes[1][i]

            # Analysis stuff
            high_analysis_images = []
            fourier_domains = []

            for channel in range(3):
                fourier_domain = np.fft.fftshift(np.fft.fft2(images[i, :, :, channel]))
                fourier_domains.append(fourier_domain)

                # Low frequency
                fourier_domain_low = fourier_domain * mask
                spatial_domain_low = np.fft.ifft2(np.fft.ifftshift(fourier_domain_low))
                image_low[:,:,channel] = np.real(spatial_domain_low)

                # High frequency
                fourier_domain_high = fourier_domain * (1 - mask)
                spatial_domain_high = np.fft.ifft2(np.fft.ifftshift(fourier_domain_high))
                image_high[:,:,channel] = np.real(spatial_domain_high)

                normal_magnitude, normal_phase = complex_to_polar_real(fourier_domain)
                high_analysis_images.extend([log_normalize(normal_magnitude), normal_phase])

                high_magnitude, high_phase = complex_to_polar_real(fourier_domain_high)
                high_analysis_images.extend([high_magnitude, high_phase])

                spatial_magnitude, _ = complex_to_polar_real(spatial_domain_high)
                high_analysis_images.append(spatial_magnitude)

            self.plot_fourier_analysis(resize_auto_interpolation_same(high_analysis_images, height, width), images_names[i])
            self.save_results(image_low, image_high, images_names[i], height, width)



    def plot_fourier_analysis(self, analysis_images: list, image_name):
        plot_images(analysis_images,
                    [
                        "[R] Normal fourier-domain mag", "[R] Normal fourier-domain phase", 
                        "[R] High fourier-domain mag", "[R] High fourier-domain phase", 
                        "[R] High spatial-domain mag",

                        "[G] Normal fourier-domain mag", "[G] Normal fourier-domain phase", 
                        "[G] High fourier-domain mag", "[G] High fourier-domain phase", 
                        "[G] High spatial-domain mag",

                        "[B] Normal fourier-domain mag", "[B] Normal fourier-domain phase", 
                        "[B] High fourier-domain mag", "[B] High fourier-domain phase", 
                        "[B] High spatial-domain mag",
                        ],
                    Path(self.high_analyze_path) / image_name,
                    cols=5)

    def save_results(self, image_low, image_high, image_name, height, width):
        image_low = resize_auto_interpolation(image_low, height, width)
        image_high = resize_auto_interpolation(image_high, height, width)

        low_path = Path(self.low_save_dir) / image_name
        high_path = Path(self.high_save_dir) / image_name

        plt.imsave(low_path, image_low.astype(np.uint8))
        plt.imsave(high_path, image_high.astype(np.uint8))

