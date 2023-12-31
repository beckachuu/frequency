import cv2
import numpy as np

from const.constants import ALIAS_RING_WIDTH


def create_radial_mask(img, radius):
    rows, cols = img.shape[:2]

    center = (cols // 2, rows // 2)
    mask = np.zeros((rows, cols))
    cv2.circle(mask, center, radius, (1, 1, 1), thickness=-1)

    # plt.imsave(Path(self.analyze_dir) / f'circle_mask_{exp_value}.png', circle_mask.astype(np.uint8))
    return mask


def create_Hann_mask(img: np.ndarray, center_intensity: int = 0):
    """
    Creates a 2D Hann window mask with the same shape as the input image.

    Parameters:
    - img (np.ndarray): Input image. Must be in HWC shape.
    - center_intensity (int): Roots to take from the 2D Hann window.
    """

    if center_intensity <= 0:
        raise ValueError("center_intensity must be positive.")

    height, width = img.shape[:2]

    # Create 1D Hann window
    hann_1d_Y = np.hanning(height)
    hann_1d_X = np.hanning(width)

    # Convert to 2D Hann window
    hann_2d = np.outer(hann_1d_Y, hann_1d_X)
    hann_2d = hann_2d**(1/center_intensity)

    return hann_2d


def create_smooth_ring_mask(img: np.ndarray, inner_radius, outer_radius, blur_strength, max_intensity):
    """
    Creates a smooth ring mask with the given inner and outer radii.

    Parameters:
    - img (np.ndarray): Input image. Must be in HWC shape.
    - inner_radius: Inner radius of the ring. Must be a non-negative number.
    - outer_radius: Outer radius of the ring. Must be a non-negative number and greater than the inner_radius.
    - blur_strength: Kernel size when applying Gaussian blur
    - max_intensity: maximum enhance value of the ring
    """

    if blur_strength <= 0 or blur_strength % 2 == 0:
        raise ValueError(f'blur_strength must be odd and positive, {blur_strength} is invalid.')

    height, width = img.shape[:2]

    scale_up = max(1, int(ALIAS_RING_WIDTH / (outer_radius - inner_radius)))
    blur_strength = blur_strength * scale_up
    if blur_strength % 2 == 0:
        blur_strength += 1

    mask_height = max(outer_radius*2, height)
    mask_width = max(outer_radius*2, width)
    center_x = mask_height // 2
    center_y = mask_width // 2

    border_x = int(abs(mask_width - width) / 2)
    border_y = int(abs(mask_height - height) / 2)

    # Create a radial gradient mask twice as large -> avoid aliasing artifact
    y, x = np.ogrid[-center_y*scale_up:center_y*scale_up, -center_x*scale_up:center_x*scale_up]
    gradient = np.sqrt(x*x + y*y)

    ring_mask = np.where((gradient > inner_radius*scale_up) & (gradient < outer_radius*scale_up), gradient, 0)

    smooth_ring = cv2.GaussianBlur(ring_mask, (blur_strength, blur_strength), 0, borderType=cv2.BORDER_CONSTANT)
    smooth_ring = cv2.resize(smooth_ring, (width, height), interpolation=cv2.INTER_AREA)
    smooth_ring = smooth_ring[border_y:border_y+height, border_x:border_x+width]
    smooth_ring = smooth_ring / np.max(smooth_ring) * max_intensity
    
    # plt.imshow(smooth_ring)
    # plt.colorbar()
    # plt.show()

    return smooth_ring

# create_smooth_ring_mask(np.zeros((640, 640)), 50, 51, 51, 1.2)
