import torch
import torch.fft
from matplotlib import pyplot as plt

from utility.path_utils import get_last_path_element


class FrequencyDomainFilter(torch.nn.Module):
    def __init__(self, filter_size):
        super().__init__()
        self.filter = torch.nn.Parameter(torch.rand(filter_size))

    def forward(self, x):
        x_fft = torch.fft.fft2(x)

        x_mag = torch.sqrt(x_fft.real**2 + x_fft.imag**2)
        x_phase = torch.atan2(x_fft.imag, x_fft.real)

        x_exp_mag = x_mag * self.filter
        x_exp = torch.complex(x_exp_mag * torch.cos(x_phase), x_exp_mag * torch.sin(x_phase))

        x_ifft = torch.fft.ifft2(x_exp)
        x_out = 2.0 * torch.sigmoid(x_ifft.real) - 0.5

        return x_out

    def save_filter_img(self, save_dir):
        # plt.imsave(save_dir, self.filter.detach().cpu().numpy())
        fig, ax = plt.subplots()
        im = ax.imshow(self.filter.detach().cpu().numpy())
        fig.colorbar(im, ax=ax)

        title = get_last_path_element(save_dir).split('.')[0]
        plt.title(title)

        plt.savefig(save_dir)
