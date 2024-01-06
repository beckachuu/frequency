import itertools
import os
from logging import Logger
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from utility.format_utils import (complex_to_polar_real, log_normalize,
                                  polar_real_to_complex,
                                  resize_auto_interpolation)
from utility.mask_util import (create_Hann_mask, create_radial_mask,
                               create_smooth_ring_mask)
from utility.path_utils import check_files_exist, create_path_if_not_exists
from utility.plot_util import plot_images


class FrequencyExp():
    def __init__(self, logger:Logger, exp_dir: str, exp_values: list, force_exp: bool, plot_analyze: bool):
        self.logger = logger

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

        inner_radii = list(np.arange(self.exp_values[0], self.exp_values[1], self.exp_values[2]))
        ring_widths = list(np.arange(self.exp_values[3], self.exp_values[4], self.exp_values[5]))
        blur_strengths = list(np.arange(self.exp_values[6], self.exp_values[7], self.exp_values[8]))
        max_intensities = list(np.arange(self.exp_values[9], self.exp_values[10], self.exp_values[11]))
        center_intensities = self.exp_values[12:]

        combinations = list(itertools.product(inner_radii, ring_widths, blur_strengths, max_intensities, center_intensities))

        for combo in combinations:
            inner_radius = int(combo[0])
            outer_radius = int(combo[0] + combo[1])
            blur_strength = self.standardize_blur_strength(combo[2])
            ring_intensity = combo[3]
            center_intensity = int(combo[4])

            self.logger.info(f"[BATCH {batch_ind}]: inner_radius = {inner_radius}, outer_radius = {outer_radius}, blur_strength = {blur_strength}, ring_intensity = {ring_intensity:.1f}, Hann_intensity = {center_intensity}")

            save_id = f'{inner_radius}-{outer_radius} {blur_strength} ring-{ring_intensity:.1f} Hann-{center_intensity}'
            ring_mask = create_smooth_ring_mask(images[0], inner_radius, outer_radius, blur_strength, ring_intensity)

            if not self.check_ring_mask(ring_mask):
                save_id = '(no corner cut) ' + save_id
            save_dir, analyze_dir = self.create_save_paths(save_id)

            if not self.force_exp and check_files_exist([Path(save_dir, image_name) for image_name in images_names]):
                self.logger.info(f'Skipping: force_exp is False and BATCH {batch_ind} has saved results for this setting.')
                continue


            hann_mask = create_Hann_mask(images[0], center_intensity)
            ring_mask = self.fill_ring_mask(ring_mask)

            mask_plot_dir = Path(analyze_dir, f'masks {save_id}.png')
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

    def check_ring_mask(self, ring_mask):
        if ring_mask[0][0] >= 1:
            self.logger.debug(f'Corners are not reduced with this setting')
            return False
        return True

    def fill_ring_mask(self, ring_mask: np.ndarray):
        # draw mask to soft-fill the ring
        diag_line = np.diag(ring_mask)
        mask_radius = abs(len(diag_line)/2 - diag_line.argmax()) * np.sqrt(2) # touches highest value of the ring
        fill = create_radial_mask(ring_mask, int(mask_radius))

        np.copyto(ring_mask, 1., where=np.logical_and(fill, ring_mask < 1))
        return ring_mask


    def standardize_blur_strength(self, blur_strength):
        if blur_strength % 2 == 0:
            blur_strength += 1
            self.logger.debug(f'blur_strength is not odd (current value: {blur_strength-1}) -> normalized to {blur_strength}')
        return int(blur_strength)
    

    def amplify_true_HFC(self, image, ring_mask, hann_mask, image_name,
                         height, width, save_dir, analyze_dir):
        
        image_exp = np.array(image)

        analyze_images = []

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
            spatial_domain = spatial_domain * hann_mask # avoid border effect after edit

            # add images to analyze
            if self.plot_analyze:
                analyze_images.append(log_normalize(magnitude))
                analyze_images.append(log_normalize(windowed_magnitude))
                analyze_images.append(log_normalize(exp_magnitude))
                spatial_magnitude, _ = complex_to_polar_real(spatial_domain)
                analyze_images.append(spatial_magnitude)

            image_exp[:,:,channel] = np.real(spatial_domain)

        image_exp = resize_auto_interpolation(image_exp, height, width)
        plt.imsave(save_dir / image_name, image_exp.astype(np.uint8))

        if self.plot_analyze:
            plot_images(analyze_images,
                        [
                            "[R] Fourier-domain mag",
                            "[R] Windowed fourier-domain mag",
                            "[R] Exp fourier-domain mag",
                            "[R] Exp spatial-domain mag",

                            "[G] Fourier-domain mag",
                            "[G] Windowed fourier-domain mag",
                            "[G] Exp fourier-domain mag",
                            "[G] Exp spatial-domain mag",

                            "[B] Fourier-domain mag",
                            "[B] Windowed fourier-domain mag",
                            "[B] Exp fourier-domain mag",
                            "[B] Exp spatial-domain mag",
                        ],
                        Path(analyze_dir) / image_name,
                        cols=4)

        return image_exp
