import os
import sys

import matplotlib.pyplot as plt
try:
    import cupy as np
    print("Running cupy with GPU")
except ImportError:
    import numpy as np
    print("Running numpy with CPU")

module_path = os.path.abspath(os.getcwd() + "/src")
if module_path not in sys.path:
    sys.path.append(module_path)

from utility.format_utils import complex_to_polar_real

plt.ioff()
plt.rcParams.update({'font.size': 20})


# Define a list of 3x3 matrices (kernels)
kernels = [
    np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]), # (vertical) prewitt
    np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]), # (horizontal) prewitt
    np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]), # (vertical) sobel
    np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]), # (horizontal) sobel
    np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]), # laplacian
    np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]), # gaussian
    np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]), # box
    np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]), # sharpen
    ]

titles = [
    '(vertical) prewitt',
    '(horizontal) prewitt',
    '(vertical) sobel',
    '(horizontal) sobel',
    'laplacian',
    'gaussian',
    'box',
    'sharpen',
]

# Create a figure
fig = plt.figure(figsize=(44, 6 * len(kernels) // 2))

for i, kernel in enumerate(kernels):
    # alter the kernel
    # kernel[0][0] += 1

    # padded_kernel = np.pad(kernel, pad_width=((0, 637), (0, 637)), mode='constant') # corner padding
    padded_kernel = np.pad(kernel, pad_width=(320, 320), mode='constant') # center padding
    fourier_transform = np.fft.fft2(padded_kernel)
    fourier_transform_shifted = np.fft.fftshift(fourier_transform)

    magnitude, phase = complex_to_polar_real(fourier_transform_shifted)

    # Plot the original kernel
    ax1 = plt.subplot(len(kernels) // 2, 6, i*3 + 1)
    im1 = ax1.imshow(kernel)
    ax1.set_title(titles[i], pad=15)
    plt.colorbar(im1, ax=ax1)

    # Plot the Phase
    ax2 = plt.subplot(len(kernels) // 2, 6, i*3 + 2)
    im2 = ax2.imshow(phase.astype(np.uint8))
    ax2.set_title('Phase {}'.format(titles[i]), pad=15)
    plt.colorbar(im2, ax=ax2)

    # Plot the Magnitude Spectrum
    ax3 = plt.subplot(len(kernels) // 2, 6, i*3 + 3)
    im3 = ax3.imshow(magnitude.astype(np.uint8))
    ax3.set_title('Magnitude {}'.format(titles[i]), pad=15)
    plt.colorbar(im3, ax=ax3)

# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.7)  # Increase the vertical spacing

plt.tight_layout()
plt.savefig('output/kernels_analyze.png', dpi=100)
plt.close(fig)
