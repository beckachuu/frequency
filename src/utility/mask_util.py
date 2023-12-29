import cv2
import numpy as np


def create_radial_mask(img, radius):
    rows, cols = img.shape[:2]

    center = (cols // 2, rows // 2)
    mask = np.zeros((rows, cols))
    cv2.circle(mask, center, radius, (1, 1, 1), thickness=-1)

    # plt.imsave(Path(self.analyze_dir) / f'circle_mask_{exp_value}.png', circle_mask.astype(np.uint8))
    return mask

