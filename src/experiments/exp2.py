import gc
import itertools
import os
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from utility.format_utils import (complex_to_polar_real, log_normalize,
                                  polar_real_to_complex,
                                  resize_auto_interpolation)
from utility.mask_util import (create_Hann_mask, create_radial_mask,
                               create_smooth_ring_mask)
from utility.mylogger import MyLogger
from utility.path_utils import create_path_if_not_exists
from utility.plot_util import plot_images


class FrequencyExp():
    def __init__(self, exp_dir: str, exp_values: list, r: int=0):
        self.r = r

        self.exp_dir = exp_dir

        self.ring_path = ''
        self.soft_ring_path = ''

        self.exp_values = exp_values


    def run_experiment(self, images, images_names, images_sizes) -> (list, list):
        '''
        Image shape must be HWC.
        '''
        logger = MyLogger.getLog()

        inner_radii = list(np.arange(self.exp_values[0], self.exp_values[1], self.exp_values[2]))
        ring_widths = list(np.arange(self.exp_values[3], self.exp_values[4], self.exp_values[5]))
        blur_strengths = list(np.arange(self.exp_values[6], self.exp_values[7], self.exp_values[8]))
        max_intensities = list(np.arange(self.exp_values[9], self.exp_values[10], self.exp_values[11]))
        center_intensities = self.exp_values[12:]

        combinations = list(itertools.product(inner_radii, ring_widths, blur_strengths, max_intensities, center_intensities))

        for combo in combinations:
            gc.collect() # avoid np.core._exceptions._ArrayMemoryError for creating and discarding large arrays frequently

            inner_radius = int(combo[0])
            outer_radius = int(combo[0] + combo[1])
            blur_strength = self.standardize_blur_strength(combo[2], logger)
            ring_intensity = combo[3]
            center_intensity = int(combo[4])

            logger.info(f"Experimenting with: inner_radius = {inner_radius}, outer_radius = {outer_radius}, blur_strength = {blur_strength}, ring_intensity = {ring_intensity:.1f}, Hann_intensity = {center_intensity}")

            hann_mask = create_Hann_mask(images[0], center_intensity)
            ring_mask = create_smooth_ring_mask(images[0], inner_radius, outer_radius, blur_strength, ring_intensity)
            if not self.check_ring_mask(ring_mask, logger):
                continue
            ring_mask = self.fill_ring_mask(ring_mask, inner_radius)
        
            save_dir = Path(self.exp_dir) / f'{inner_radius}-{outer_radius} {blur_strength} ring-{ring_intensity:.1f} Hann-{center_intensity}'
            create_path_if_not_exists(save_dir)


            mask_plot_dir = Path(self.exp_dir, 
                                 f'masks {inner_radius}-{outer_radius} {blur_strength} ring-{ring_intensity:.1f} Hann-{center_intensity}.png')

            # if not os.path.isfile(mask_plot_dir):
            plot_images([ring_mask, hann_mask], [f'ring_mask blur-{blur_strength} intense-{ring_intensity:.1f}', f'hann_mask {center_intensity}'],
                        mask_plot_dir)
            for i in range(images.shape[0]):
                # logger.info(f"Experimenting on image: {images_names[i]}")

                height = int(images_sizes[0][i])
                width = int(images_sizes[1][i])
                
                exp_image = self.amplify_true_HFC(images[i], ring_mask, hann_mask, images_names[i], 
                                        height, width, save_dir)

                # plot_images([
                                
                #             ],
                #         [
                #             "[R] Normal fourier-domain mag",
                #             "[R] Exp fourier-domain mag",
                #             "[R] Exp spatial-domain mag",

                #             "[G] Normal fourier-domain mag",
                #             "[G] Exp fourier-domain mag",
                #             "[G] Exp spatial-domain mag",

                #             "[B] Normal fourier-domain mag",
                #             "[B] Exp fourier-domain mag",
                #             "[B] Exp spatial-domain mag",
                #         ],
                #     Path(save_dir) / images_names[i],
                #     cols=3)
    

    def check_ring_mask(self, ring_mask, logger: MyLogger):
        if ring_mask[0][0] >= 1:
            logger.warning(f'Ring mask does not work with this setting: corners are not reduced -> skipping to next set of values')
            return False
        return True

    def fill_ring_mask(self, ring_mask, inner_radius):
        fill = create_radial_mask(ring_mask, inner_radius)
        np.copyto(ring_mask, 1., where=np.logical_and(fill, ring_mask < 1))
        return ring_mask


    def standardize_blur_strength(self, blur_strength, logger: MyLogger):
        if blur_strength % 2 == 0:
            blur_strength += 1
            logger.debug(f'blur_strength is not odd (current value: {blur_strength-1}) -> normalized to {blur_strength}')
        return int(blur_strength)
    

    def amplify_true_HFC(self, image, ring_mask, hann_mask, image_name,
                         height, width, save_dir):
        
        image_exp = np.array(image)

        for channel in range(3):
            windowed_image = image[:, :, channel] * hann_mask
            fourier_domain = np.fft.fftshift(np.fft.fft2(windowed_image))

            magnitude, phase = complex_to_polar_real(fourier_domain)

            # np.copyto(magnitude, magnitude * reduce_corner_mask, where=reduce_corner_mask.astype(bool))
            # np.copyto(magnitude, magnitude * amplify_ring_mask, where=amplify_ring_mask.astype(bool))

            magnitude = magnitude * ring_mask

            complex = polar_real_to_complex(magnitude, phase)
            spatial_domain = np.fft.ifft2(np.fft.ifftshift(complex))
            image_exp[:,:,channel] = np.real(spatial_domain)

        image_exp = resize_auto_interpolation(image_exp, height, width)
        plt.imsave(save_dir / image_name, image_exp.astype(np.uint8))

        return image_exp
