import itertools
import os
import sys
from logging import Logger
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

module_path = os.path.abspath(os.getcwd() + "/src")
if module_path not in sys.path:
    sys.path.append(module_path)


from utility.format_utils import (complex_to_polar_real, polar_real_to_complex,
                                  resize_auto_interpolation)
from utility.mask_util import create_radial_mask
from utility.path_utils import check_files_exist, create_path_if_not_exists


class FrequencyExp():
    def __init__(self, logger:Logger, exp_dir: str, exp_values: list, force_exp: bool, plot_analyze: bool):
        self.logger = logger

        self.exp_dir = exp_dir

        self.phase_path = ''
        self.magnitude_path = ''
        self.both_path = ''

        self.exp_values = exp_values
        self.force_exp = force_exp
        self.plot_analyze = plot_analyze
    

    def run_experiment(self, batch_ind, images, images_names, images_sizes) -> (list, list):
        '''
        Image shape must be HWC.
        '''

        radii = list(np.arange(self.exp_values[0], self.exp_values[1], self.exp_values[2]))
        alphas = list(np.arange(self.exp_values[3], self.exp_values[4], self.exp_values[5]))

        combinations = list(itertools.product(radii, alphas))

        for combo in combinations:
            radius = int(combo[0])
            alpha = round(combo[1], 2)

            mask = create_radial_mask(np.zeros([images.shape[1], images.shape[2]]), radius)
        
            self.logger.info(f"[BATCH {batch_ind}]: exp_value = {alpha}")

            self.phase_path = Path(self.exp_dir) / f'hybrid_phase-{alpha}'
            self.magnitude_path = Path(self.exp_dir) / f'hybrid_magnitude-{alpha}'
            self.both_path = Path(self.exp_dir) / f'hybrid_both-{alpha}'

            images_list_check = [Path(self.phase_path, image_name) for image_name in images_names]
            images_list_check += [Path(self.magnitude_path, image_name) for image_name in images_names]
            images_list_check += [Path(self.both_path, image_name) for image_name in images_names]
            
            create_path_if_not_exists(self.phase_path)
            create_path_if_not_exists(self.magnitude_path)
            create_path_if_not_exists(self.both_path)

            for i in range(len(images)):
                if not self.force_exp and check_files_exist(images_list_check):
                    self.logger.info(f'Skipping: force_exp is False and BATCH {batch_ind} has saved results for this setting.')
                    continue

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
                

                self.support_channel_info(alpha, images[i], images_names[i], height, width, fourier_domains, high_magnitudes, high_phases)



    def support_channel_info(self, alpha, image, image_name, height, width, fourier_domains, high_magnitudes, high_phases):
        hybrid_phase = np.array(image)
        hybrid_magnitude = np.array(image)
        hybrid_both = np.array(image)

        for channel in range(3):
            magnitude, phase = complex_to_polar_real(fourier_domains[channel])
            new_magnitude = magnitude \
                + alpha/2 * (high_magnitudes[channel - 1] - high_magnitudes[channel]) \
                    + alpha/2 * (high_magnitudes[channel - 2] - high_magnitudes[channel])
            new_phase = phase \
                + alpha/2 * (high_phases[channel-1] - high_phases[channel]) \
                    + alpha/2 * (high_phases[channel-2] - high_phases[channel])

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

