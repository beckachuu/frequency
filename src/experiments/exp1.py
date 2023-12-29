import os
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

module_path = os.path.abspath(os.getcwd() + "/src")
if module_path not in sys.path:
    sys.path.append(module_path)


from utility.format_utils import (complex_to_polar_real, polar_real_to_complex,
                                  resize_auto_interpolation)
from utility.mask_util import create_radial_mask
from utility.mylogger import MyLogger
from utility.path_utils import create_path_if_not_exists


class FrequencyExp():
    def __init__(self, exp_dir: str, exp_values: list, r: int=0):
        self.r = r

        self.exp_dir = exp_dir

        self.phase_path = ''
        self.magnitude_path = ''
        self.both_path = ''

        self.exp_values = exp_values
    

    def run_experiment(self, images, images_names, images_sizes) -> (list, list):
        '''
        Image shape must be HWC.
        '''
        logger = MyLogger.getLog()

        mask = create_radial_mask(np.zeros([images.shape[1], images.shape[2]]), self.r)

        for exp_value in self.exp_values:

            self.phase_path = Path(self.exp_dir) / f'hybrid_phase-{exp_value}'
            self.magnitude_path = Path(self.exp_dir) / f'hybrid_magnitude-{exp_value}'
            self.both_path = Path(self.exp_dir) / f'hybrid_both-{exp_value}'
            create_path_if_not_exists(self.phase_path)
            create_path_if_not_exists(self.magnitude_path)
            create_path_if_not_exists(self.both_path)

            for i in range(images.shape[0]):
                logger.info(f"Experimenting on image: {images_names[i]}")

                image_low = np.zeros([images.shape[1], images.shape[2], 3])
                image_high = np.zeros([images.shape[1], images.shape[2], 3])
                height = int(images_sizes[0][i])
                width = int(images_sizes[1][i])

                fourier_domains = []
                high_magnitudes = []
                high_phases = []

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

                    high_magnitude, high_phase = complex_to_polar_real(fourier_domain_high)
                    high_magnitudes.append(high_magnitude)
                    high_phases.append(high_phase)
                

                self.support_channel_info(exp_value, images[i], images_names[i], height, width, fourier_domains, high_magnitudes, high_phases)


                # plot_images([
                                
                #             ],
                #         [
                #             "[R] Normal fourier-domain mag", "[R] Normal fourier-domain phase", 
                #             "[R] High fourier-domain mag", "[R] High fourier-domain phase", 
                #             "[R] High spatial-domain mag",

                #             "[G] Normal fourier-domain mag", "[G] Normal fourier-domain phase", 
                #             "[G] High fourier-domain mag", "[G] High fourier-domain phase", 
                #             "[G] High spatial-domain mag",

                #             "[B] Normal fourier-domain mag", "[B] Normal fourier-domain phase", 
                #             "[B] High fourier-domain mag", "[B] High fourier-domain phase", 
                #             "[B] High spatial-domain mag",
                #         ],
                #     Path(self.analyze_dir) / images_names[i],
                #     cols=5)



    def support_channel_info(self, exp_value, image, image_name, height, width, fourier_domains, high_magnitudes, high_phases):
        hybrid_phase = np.array(image)
        hybrid_magnitude = np.array(image)
        hybrid_both = np.array(image)

        for channel in range(3):
            magnitude, phase = complex_to_polar_real(fourier_domains[channel])
            new_magnitude = magnitude \
                + exp_value/2 * (high_magnitudes[channel - 1] - high_magnitudes[channel]) \
                    + exp_value/2 * (high_magnitudes[channel - 2] - high_magnitudes[channel])
            new_phase = phase \
                + exp_value/2 * (high_phases[channel-1] - high_phases[channel]) \
                    + exp_value/2 * (high_phases[channel-2] - high_phases[channel])

            complex = polar_real_to_complex(magnitude, new_phase)
            spatial_domain = np.fft.ifft2(np.fft.ifftshift(complex))
            hybrid_phase[:,:,channel] = np.real(spatial_domain)
            
            complex = polar_real_to_complex(new_magnitude, phase)
            spatial_domain = np.fft.ifft2(np.fft.ifftshift(complex))
            hybrid_magnitude[:,:,channel] = np.real(spatial_domain)

            complex = polar_real_to_complex(new_magnitude, new_phase)
            spatial_domain = np.fft.ifft2(np.fft.ifftshift(complex))
            hybrid_both[:,:,channel] = np.real(spatial_domain)

        hybrid_phase = resize_auto_interpolation(hybrid_phase, height, width)
        hybrid_magnitude = resize_auto_interpolation(hybrid_magnitude, height, width)
        hybrid_both = resize_auto_interpolation(hybrid_both, height, width)

        plt.imsave(self.phase_path / image_name, hybrid_phase.astype(np.uint8))
        plt.imsave(self.magnitude_path / image_name, hybrid_magnitude.astype(np.uint8))
        plt.imsave(self.both_path / image_name, hybrid_both.astype(np.uint8))

