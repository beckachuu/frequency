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
    def __init__(self, logger:Logger, input_dir, image_extensions, batch_size, exp_dir: str, exp_values: list, 
                 force_exp: bool, plot_analyze: bool):
        self.logger = logger

        self.input_dir = input_dir
        self.image_extensions = image_extensions
        self.batch_size = batch_size
        self.exp_dir = exp_dir
        self.exp_values = exp_values
        self.force_exp = force_exp
        self.plot_analyze = plot_analyze

        self.ring_path = ''
        self.soft_ring_path = ''


    def run_experiment(self):
        dataset = ImageDataset(self.input_dir, self.image_extensions)
        dataloader = DataLoader(dataset=dataset, num_workers=2, batch_size=self.batch_size, shuffle=True)

        self.logger.info(f'Input path: {self.input_dir} (images count: {len(dataset)}).')

        for batch_ind, (images, images_names, images_sizes) in enumerate(dataloader):

            inner_radii = list(np.arange(self.exp_values[0], self.exp_values[1] + self.exp_values[2], self.exp_values[2]))
            ring_widths = list(np.arange(self.exp_values[3], self.exp_values[4] + self.exp_values[5], self.exp_values[5]))
            blur_strengths = list(np.arange(self.exp_values[6], self.exp_values[7] + self.exp_values[8], self.exp_values[8]))
            max_intensities = list(np.arange(self.exp_values[9], self.exp_values[10] + self.exp_values[11], self.exp_values[11]))
            center_intensities = self.exp_values[12:]

            square_h, square_w = images[0].shape[:2]
            big_img = np.zeros((square_h * 3, square_w * 3))
            big_h, big_w = big_img.shape[:2]

            combinations = list(itertools.product(inner_radii, ring_widths, blur_strengths, max_intensities, center_intensities))

            for combo in combinations:
                half_size =  max(big_h, big_w) / 2
                inner_radius = int(combo[0] * half_size)
                outer_radius = int(combo[0] + combo[1] * half_size)
                blur_strength = self.standardize_blur_strength(combo[2] * half_size)
                ring_enhance = combo[3]
                hann_intensity = int(combo[4])

                self.logger.info(f"[BATCH {batch_ind}]: inner_radius = {inner_radius}, outer_radius = {outer_radius}, blur_strength = {blur_strength}, ring_intensity = {ring_enhance:.1f}, Hann_intensity = {hann_intensity}")

                save_id = f'{inner_radius}-{outer_radius} {blur_strength} ring-{ring_enhance:.1f} Hann-{hann_intensity}'
                ring_mask = create_smooth_ring_mask(big_h, big_w, inner_radius, outer_radius, blur_strength, ring_enhance)

                if not self.check_ring_mask(ring_mask):
                    save_id = '(no corner cut) ' + save_id
                save_dir = Path(self.exp_dir, save_id)
                create_path_if_not_exists(save_dir)

                if not self.force_exp and check_files_exist([Path(save_dir, image_name) for image_name in images_names]):
                    self.logger.info(f'Skipping: force_exp is False and BATCH {batch_ind} has saved results for this setting.')
                    continue

                hann_mask = create_Hann_mask(big_h, big_w, hann_intensity)
                ring_mask = self.fill_ring_mask(ring_mask)

                create_path_if_not_exists(self.exp_dir)
                plt.imsave(Path(self.exp_dir, f'ring_mask blur-{blur_strength} intense-{ring_enhance:.1f}.png'), to_numpy(ring_mask))
                if hann_intensity > 0:
                    plt.imsave(Path(self.exp_dir, f'hann_mask {hann_intensity}.png'), to_numpy(hann_mask))

                analyze_dir = Path(save_dir) / 'fourier_plots'
                create_path_if_not_exists(analyze_dir)

                for i in range(images.shape[0]):
                    img_h = int(images_sizes[0][i])
                    img_w = int(images_sizes[1][i])
                    
                    self.amplify_true_HFC(images[i], big_img, ring_mask, hann_mask, images_names[i], 
                                            square_h, square_w, img_h, img_w, save_dir, analyze_dir)

    

    def check_ring_mask(self, ring_mask):
        if ring_mask[0][0] >= 1:
            self.logger.debug(f'Corners are not reduced with this setting')
            return False
        return True

    def fill_ring_mask(self, ring_mask: np.ndarray):
        # draw mask to soft-fill the ring
        diag_line = np.diag(ring_mask)
        mask_radius = abs(len(diag_line)/2 - diag_line.argmax()) * np.sqrt(2) # touches highest value of the ring
        if mask_radius < 1:
            return ring_mask
        h, w = ring_mask.shape[:2]
        fill = create_radial_mask(h, w, int(mask_radius))

        np.copyto(ring_mask, 1., where=np.logical_and(fill, ring_mask < 1))
        return ring_mask


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
