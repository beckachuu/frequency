import itertools
from logging import Logger
from pathlib import Path

from torch.utils.data import DataLoader

try:
    import cupy as np
    print("Running cupy with GPU")
except ImportError:
    import numpy as np
    print("Running numpy with CPU")
from matplotlib import pyplot as plt

from dataset import ImageDataset
from utility.format_utils import (complex_to_polar_real, crop_center,
                                  log_normalize, polar_real_to_complex,
                                  resize_auto_interpolation, to_cupy, to_numpy)
from utility.mask_util import (create_Hann_mask, create_radial_mask,
                               create_smooth_ring_mask, smooth_edges)
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


    def run_experiment(self, input_dir, image_extensions, batch_size):
        dataset = ImageDataset(input_dir, image_extensions)
        dataloader = DataLoader(dataset=dataset, num_workers=2, batch_size=batch_size, shuffle=True)

        self.logger.info(f'Input count: {len(dataset)}.')

        for batch_ind, (images, images_names, images_sizes) in enumerate(dataloader):

            inner_radii = self.exp_values[0]
            outer_radii = self.exp_values[1]
            blur_strengths = self.exp_values[2]
            ring_enhances = self.exp_values[3]
            hann_intensities = self.exp_values[4]

            square_h, square_w = images[0].shape[:2]
            big_img = np.zeros((square_h * 3, square_w * 3))
            big_h, big_w = big_img.shape[:2]

            combinations = list(itertools.product(range(len(inner_radii)), blur_strengths, ring_enhances, hann_intensities))

            for combo in combinations:
                half_size =  max(big_h, big_w) / 2

                inner_radius = int(inner_radii[combo[0]] * half_size)
                outer_radius = int(outer_radii[combo[0]] * half_size)
                blur_strength = self.standardize_blur_strength(combo[1] * half_size)
                ring_enhance = combo[2]
                hann_intensity = int(combo[3])

                self.logger.info(f"[BATCH {batch_ind}]: inner_radius = {inner_radius}, outer_radius = {outer_radius}, blur_strength = {blur_strength}, ring_intensity = {ring_enhance:.1f}, Hann_intensity = {hann_intensity}")

                save_id = f'{inner_radius}-{outer_radius} {blur_strength} ring-{ring_enhance:.1f} Hann-{hann_intensity}'
                save_dir = Path(self.exp_dir, save_id)
                create_path_if_not_exists(save_dir)

                hann_mask, ring_mask = self.create_masks(big_h, big_w, inner_radius, outer_radius, blur_strength, 
                                                         hann_intensity, ring_enhance)

                if self.check_continue(save_dir, images_names, ring_mask):
                    continue

                analyze_dir = Path(save_dir) / 'fourier_plots'
                create_path_if_not_exists(analyze_dir)

                for i in range(images.shape[0]):
                    img_h = int(images_sizes[0][i])
                    img_w = int(images_sizes[1][i])
                    
                    self.amplify_true_HFC(images[i], big_img, ring_mask, hann_mask, images_names[i], 
                                            square_h, square_w, img_h, img_w, save_dir, analyze_dir)


    def create_masks(self, big_h, big_w, inner_radius, outer_radius, blur_strength, hann_intensity, ring_enhance):
        hann_mask = create_Hann_mask(big_h, big_w, hann_intensity)
        ring_mask = create_smooth_ring_mask(big_h, big_w, inner_radius, outer_radius, blur_strength, ring_enhance)
        ring_mask = self.reduce_ring_mask(ring_mask, ring_enhance)

        create_path_if_not_exists(self.exp_dir)
        plt.imsave(Path(self.exp_dir, f'ring_mask {inner_radius}-{outer_radius} blur-{blur_strength} intense-{ring_enhance:.1f}.png'), to_numpy(ring_mask))
        if hann_intensity > 0:
            plt.imsave(Path(self.exp_dir, f'hann_mask {hann_intensity}.png'), to_numpy(hann_mask))

        return hann_mask, ring_mask

    
    def check_continue(self, save_dir, images_names, ring_mask):
        if ring_mask.max() > 1:
            raise ValueError('ring_enhance value must be in range [0, 1].')

        if not self.force_exp and check_files_exist([Path(save_dir, image_name) for image_name in images_names]):
            self.logger.info(f'Skipping: force_exp is False and this batch has saved results.')
            return True
        
        return False


    def reduce_ring_mask(self, ring_mask: np.ndarray, ring_enhance):
        # draw mask to soft-fill the ring
        diag_line = np.diag(ring_mask)
        mask_radius = abs(len(diag_line)/2 - diag_line.argmax()) * np.sqrt(2) # touches highest value of the ring
        if mask_radius >= 1:
            h, w = ring_mask.shape[:2]
            fill = create_radial_mask(h, w, int(mask_radius))

            np.copyto(ring_mask, ring_enhance, where=np.logical_and(fill, ring_mask < ring_enhance))
        
        result_ring = np.ones(ring_mask.shape) - ring_mask
        
        return result_ring


    def standardize_blur_strength(self, blur_strength):
        if blur_strength % 2 == 0:
            blur_strength += 1
            self.logger.debug(f'blur_strength is not odd (current value: {blur_strength-1}) -> normalized to {blur_strength}')
        return int(blur_strength)
    

    def amplify_true_HFC(self, image, big_img, ring_mask, hann_mask, image_name,
                         square_h, square_w, img_h, img_w, save_dir, analyze_dir):
        
        image_exp = np.array(image)

        analyze_images = []

        for channel in range(3):
            image_channel = to_cupy(image[:, :, channel])
            fourier_domain = np.fft.fftshift(np.fft.fft2(image_channel))
            magnitude, _ = complex_to_polar_real(fourier_domain)

            # smooth edges to avoid edge effects
            windowed_image = smooth_edges(image_channel, big_img, hann_mask)
            windowed_fourier_domain = np.fft.fftshift(np.fft.fft2(windowed_image))
            windowed_magnitude, windowed_phase = complex_to_polar_real(windowed_fourier_domain)

            # apply ring enhance mask
            exp_magnitude = windowed_magnitude * ring_mask
            complex = polar_real_to_complex(exp_magnitude, windowed_phase)
            exp_spatial_domain = np.fft.ifft2(np.fft.ifftshift(complex))
            exp_spatial_domain = crop_center(exp_spatial_domain, square_h, square_w)

            # add images to analyze
            if self.plot_analyze:
                analyze_images.append(to_numpy(log_normalize(magnitude)))
                analyze_images.append(to_numpy(log_normalize(windowed_magnitude)))
                analyze_images.append(to_numpy(log_normalize(exp_magnitude)))
                spatial_magnitude, _ = complex_to_polar_real(exp_spatial_domain)
                analyze_images.append(to_numpy(spatial_magnitude))

            image_exp[:,:,channel] = np.clip(np.real(exp_spatial_domain), 0, 255)

        image_exp = resize_auto_interpolation(image_exp, img_h, img_w)
        plt.imsave(save_dir / image_name, to_numpy(image_exp.astype(np.uint8)))

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
