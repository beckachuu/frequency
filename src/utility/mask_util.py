import cv2
import numpy as np

from const.constants import ALIAS_RING_WIDTH


def create_radial_mask(height: int, width: int, radius: int):
    if radius < 0:
        raise ValueError("Radius must be non-negative.")
    
    center = (width // 2, height // 2)
    mask = np.zeros((height, width))
    cv2.circle(mask, center, radius, (1, 1, 1), thickness=-1)

    # plt.imsave(Path(self.analyze_dir) / f'circle_mask_{exp_value}.png', circle_mask.astype(np.uint8))
    return mask


def create_Hann_mask(height: int, width: int, center_intensity: int = 0):
    """
    Creates a 2D Hann window mask with the same shape as the input image.

    Parameters:
    - height, width: mask size
    - center_intensity (int): Roots to take from the 2D Hann window.
    """

    if center_intensity <= 0:
        return(np.ones((height, width)))

    # Create 1D Hann window
    hann_1d_Y = np.hanning(height)
    hann_1d_X = np.hanning(width)

    # Convert to 2D Hann window
    hann_2d = np.outer(hann_1d_Y, hann_1d_X)
    hann_2d = hann_2d**(1/center_intensity)

    return hann_2d


def create_smooth_ring_mask(height: int, width: int, inner_radius: float, outer_radius, blur_strength, max_enhance):
    """
    Creates a smooth ring mask with the given inner and outer radii.

    Parameters:
    - height, width: mask size
    - inner_radius: Inner radius of the ring. Must be a non-negative number.
    - outer_radius: Outer radius of the ring. Must be a non-negative number and greater than the inner_radius.
    - blur_strength: Kernel size when applying Gaussian blur
    - max_intensity: maximum enhance value of the ring
    """

    if blur_strength <= 0 or blur_strength % 2 == 0:
        raise ValueError(f'blur_strength must be odd and positive, {blur_strength} is invalid.')

    # Scale image up if ring width is too thin -> avoid artifacts on thin ring
    scale_up = max(1, int(ALIAS_RING_WIDTH / (outer_radius - inner_radius)))
    blur_strength = blur_strength * scale_up
    if blur_strength % 2 == 0:
        blur_strength += 1

    # Create a radial gradient mask as large as the ring width -> avoid border issue when applying Gaussian blur
    mask_height = max(outer_radius*2, height)
    mask_width = max(outer_radius*2, width)
    center_x = mask_height // 2
    center_y = mask_width // 2

    border_x = abs(mask_width - width) // 2
    border_y = abs(mask_height - height) // 2

    y, x = np.ogrid[-center_y*scale_up:center_y*scale_up, -center_x*scale_up:center_x*scale_up]
    gradient = np.sqrt(x*x + y*y)

    ring_mask = np.where((gradient > inner_radius*scale_up) & (gradient < outer_radius*scale_up), gradient, 0)

    smooth_ring = cv2.GaussianBlur(ring_mask, (blur_strength, blur_strength), 0, borderType=cv2.BORDER_CONSTANT)
    smooth_ring = cv2.resize(smooth_ring, (mask_width, mask_height), interpolation=cv2.INTER_AREA) # resize to padded size
    smooth_ring = smooth_ring[border_y:border_y+height, border_x:border_x+width]
    smooth_ring = smooth_ring / np.max(smooth_ring) * max_enhance
    
    # plt.imshow(smooth_ring)
    # plt.colorbar()
    # plt.show()

    return smooth_ring


def smooth_edges(image: np.ndarray, big_img: np.ndarray, hann_mask=None) -> np.ndarray:
    image = np.array(image)
    if hann_mask is None:
        hann_mask = create_Hann_mask(big_img, 1)
    
    if image.ndim == 3:
        height, width, channels = image.shape
    else:
        height, width = image.shape
        channels = 1

    for channel in range(channels):
        if channels == 1:
            channel_image = image
        else:
            channel_image = image[:, :, channel]

        flip_h = np.flip(channel_image, 1)
        flip_v = np.flip(channel_image, 0)
        flip_both = np.flip(channel_image, (0, 1))
        # TODO: handle when big_img is not exactly 3 times larger
        if big_img.ndim == 3:
            big_img[height:2*height,width:2*width,channel] = channel_image
            big_img[:height,width:2*width,channel] = big_img[2*height:3*height,width:2*width,channel] = flip_h
            big_img[height:2*height,:width,channel] = big_img[height:2*height,2*width:3*width,channel] = flip_v
            big_img[:height,:width,channel] = big_img[2*height:3*height,:width,channel] = flip_both
            big_img[:height,2*width:3*width,channel] = big_img[2*height:3*height,2*width:3*width,channel] = flip_both
        else:
            big_img[height:2*height,width:2*width] = channel_image
            big_img[:height,width:2*width] = big_img[2*height:3*height,width:2*width] = flip_h
            big_img[height:2*height,:width] = big_img[height:2*height,2*width:3*width] = flip_v
            big_img[:height,:width] = big_img[2*height:3*height,:width] = flip_both
            big_img[:height,2*width:3*width] = big_img[2*height:3*height,2*width:3*width] = flip_both

        big_img = big_img * hann_mask
    return big_img
# plot_images([create_Hann_mask(np.zeros((640, 640)), i) for i in [4, 5, 6, 7]], ['4', '5', '6', '7'], 'test_Hann.png')
