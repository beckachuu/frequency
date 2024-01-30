import torch
import torch.fft
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

from const.constants import PLOT_FONT_SCALE, RGB_CHANNELS_NAMES


class FrequencyDomainFilter(torch.nn.Module):
    def __init__(self, filter_size, num_channels=3):
        super().__init__()
        self.filter = torch.nn.Parameter(torch.rand(num_channels, *filter_size))

    def forward(self, x):
        _, channels, _, _ = x.shape # BCHW
        num_channels = self.filter.shape[0]

        x_out = torch.zeros_like(x)

        for i in range(channels):
            x_channel = x[:, i]

            x_fft = torch.fft.fft2(x_channel)

            x_real = torch.view_as_real(x_fft)
            x_mag = torch.sqrt(x_real[..., 0]**2 + x_real[..., 1]**2)
            x_phase = torch.atan2(x_real[..., 1], x_real[..., 0])

            x_exp_mag = x_mag * self.filter[i % num_channels]
            x_exp = torch.view_as_complex(torch.stack((x_exp_mag * torch.cos(x_phase), x_exp_mag * torch.sin(x_phase)), dim=-1))

            x_ifft = torch.fft.ifft2(x_exp)
            x_out[:, i] = 2.0 * torch.sigmoid(x_ifft.real) - 0.5

        return x_out


    def save_filter_img(self, save_dir, dpi=100):
        num_channels = self.filter.shape[0]
        size = self.filter.shape[1]
        font_size = size / PLOT_FONT_SCALE

        plt.rcParams.update({'font.size': font_size})

        fig_width = size * num_channels / dpi
        fig_height = size / dpi
        fig, axs = plt.subplots(1, num_channels, figsize=(fig_width, fig_height), dpi=dpi)

        for i in range(num_channels):
            im = axs[i].imshow(self.filter[i].detach().cpu().numpy())
            if num_channels == len(RGB_CHANNELS_NAMES):
                axs[i].set_title(RGB_CHANNELS_NAMES[i])
            else:
                axs[i].set_title(f'Channel {i}')

            divider = make_axes_locatable(axs[i])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(im, cax=cax, format=FormatStrFormatter('%.1f'))
            cbar.ax.tick_params(labelsize=font_size)  # Adjust font size here

        plt.tight_layout()
        plt.savefig(save_dir)
        plt.close(fig)

