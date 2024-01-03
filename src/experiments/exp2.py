import gc
import itertools
import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from utility.format_utils import (complex_to_polar_real, log_normalize,
                                  polar_real_to_complex,
                                  resize_auto_interpolation)
from utility.mask_util import (create_Hann_mask, create_radial_mask,
                               create_smooth_ring_mask)
from utility.mylogger import MyLogger
from utility.path_utils import check_files_exist, create_path_if_not_exists
from utility.plot_util import plot_images


class FrequencyExp():
    def __init__(self, exp_dir: str, exp_values: list, force_exp: bool, plot_analyze: bool):
        self.exp_dir = exp_dir

        self.ring_path = ''
        self.soft_ring_path = ''

        self.exp_values = exp_values
        self.force_exp = force_exp
        self.plot_analyze = plot_analyze


    def run_experiment(self, batch_ind, images, images_names, images_sizes) -> (list, list):
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
            inner_radius = int(combo[0])
            outer_radius = int(combo[0] + combo[1])
            blur_strength = self.standardize_blur_strength(combo[2], logger)
            ring_intensity = combo[3]
            center_intensity = int(combo[4])

            logger.info(f"[BATCH {batch_ind}]: inner_radius = {inner_radius}, outer_radius = {outer_radius}, blur_strength = {blur_strength}, ring_intensity = {ring_intensity:.1f}, Hann_intensity = {center_intensity}")

            hann_mask = create_Hann_mask(images[0], center_intensity)
            ring_mask = create_smooth_ring_mask(images[0], inner_radius, outer_radius, blur_strength, ring_intensity)
            ring_mask = self.fill_ring_mask(ring_mask)

            save_id = f'{inner_radius}-{outer_radius} {blur_strength} ring-{ring_intensity:.1f} Hann-{center_intensity}'
            if not self.check_ring_mask(ring_mask, logger):
                save_id = '(no corner cut) ' + save_id
            save_dir, analyze_dir = self.create_save_paths(save_id)

            if not self.force_exp and check_files_exist([Path(save_dir, image_name) for image_name in images_names]):
                logger.info(f'Skipping: force_exp is False and BATCH {batch_ind} has saved results for this setting.')
                continue

            mask_plot_dir = Path(self.exp_dir, f'masks {save_id}.png')
            if self.force_exp or not os.path.isfile(mask_plot_dir):
                plot_images([ring_mask, hann_mask], [f'ring_mask blur-{blur_strength} intense-{ring_intensity:.1f}', f'hann_mask {center_intensity}'],
                            mask_plot_dir)

            for i in range(images.shape[0]):
                height = int(images_sizes[0][i])
                width = int(images_sizes[1][i])
                
                self.amplify_true_HFC(images[i], ring_mask, hann_mask, images_names[i], 
                                        height, width, save_dir, analyze_dir)
    

    def create_save_paths(self, save_id):
        save_dir = Path(self.exp_dir, save_id)
        create_path_if_not_exists(save_dir)
        analyze_dir = Path(save_dir) / 'fourier_plots'
        create_path_if_not_exists(analyze_dir)
        return save_dir, analyze_dir

    def check_ring_mask(self, ring_mask, logger: MyLogger):
        if ring_mask[0][0] >= 1:
            logger.debug(f'Corners are not reduced with this setting')
            return False
        return True

    def fill_ring_mask(self, ring_mask: np.ndarray):
        # draw mask to soft-fill the ring
        mid_row = ring_mask[int(ring_mask.shape[0]/2)]
        mask_radius = abs(int(len(mid_row)/2) - mid_row.argmax()) # touches highest value of the ring
        fill = create_radial_mask(ring_mask, mask_radius)

        np.copyto(ring_mask, 1., where=np.logical_and(fill, ring_mask < 1))
        return ring_mask


    def standardize_blur_strength(self, blur_strength, logger: MyLogger):
        if blur_strength % 2 == 0:
            blur_strength += 1
            logger.debug(f'blur_strength is not odd (current value: {blur_strength-1}) -> normalized to {blur_strength}')
        return int(blur_strength)
    

    def amplify_true_HFC(self, image, ring_mask, hann_mask, image_name,
                         height, width, save_dir, analyze_dir):
        
        image_exp = np.array(image)

        for channel in range(3):
            fourier_domain = np.fft.fftshift(np.fft.fft2(image[:, :, channel]))
            magnitude, phase = complex_to_polar_real(fourier_domain)

            # apply Hann window
            windowed_image = image[:, :, channel] * hann_mask
            windowed_fourier_domain = np.fft.fftshift(np.fft.fft2(windowed_image))
            windowed_magnitude, _ = complex_to_polar_real(windowed_fourier_domain)

            # apply ring enhance mask
            exp_magnitude = windowed_magnitude * ring_mask
            complex = polar_real_to_complex(exp_magnitude, phase)
            spatial_domain = np.fft.ifft2(np.fft.ifftshift(complex))
            spatial_domain = spatial_domain * hann_mask

            image_exp[:,:,channel] = np.real(spatial_domain) # avoid border effect after edit

        image_exp = resize_auto_interpolation(image_exp, height, width)
        plt.imsave(save_dir / image_name, image_exp.astype(np.uint8))

        return image_exp
