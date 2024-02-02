import torch
import torch.fft
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

from const.constants import PLOT_FONT_SCALE, RGB_CHANNELS_NAMES


class FrequencyDomainFilter(torch.nn.Module):
    def __init__(self, filter_size: tuple, num_channels=3):
        super().__init__()
        assert isinstance(filter_size, tuple) and len(filter_size) == 2, "filter_size must be a tuple of two elements"
        assert isinstance(num_channels, int) and num_channels > 0, "num_channels must be a positive integer"

        self.one_sided_filter = torch.nn.Parameter(torch.rand(num_channels, *(filter_size[0] // 2, filter_size[1])))
        self.batch_norm = torch.nn.BatchNorm2d(num_channels)

    def forward(self, x):
        _, channels, _, _ = x.shape # BCHW

        # Ensure symmetric for filter
        symmetric_part = torch.flip(self.one_sided_filter, dims=[-1, -2])
        full_filter = torch.cat([self.one_sided_filter, symmetric_part], dim=-2)

        x_fft = torch.fft.fft2(x)

        x_real = torch.view_as_real(x_fft)
        x_mag = torch.sqrt(x_real[..., 0]**2 + x_real[..., 1]**2)
        x_phase = torch.atan2(x_real[..., 1], x_real[..., 0])

        x_exp_mag = x_mag * full_filter[:channels]
        x_exp = torch.view_as_complex(torch.stack((x_exp_mag * torch.cos(x_phase), x_exp_mag * torch.sin(x_phase)), dim=-1))
        x_ifft = torch.fft.ifft2(x_exp)

        x_out = self.batch_norm(x_ifft.real)
        x_out = 2.0 * torch.sigmoid(x_out) - 0.5

        return x_out

    def create_full_filter(self):
        symmetric_part = torch.flip(self.one_sided_filter, dims=[-1, -2])
        full_filter = torch.cat([self.one_sided_filter, symmetric_part], dim=-2)
        return full_filter


    def save_filter_img(self, save_dir, dpi=100):
        symmetric_filter = self.create_full_filter()
        num_channels = symmetric_filter.shape[0]
        size = symmetric_filter.shape[1]
        font_size = size / PLOT_FONT_SCALE

        plt.rcParams.update({'font.size': font_size})

        fig_width = size * num_channels / dpi
        fig_height = size / dpi
        fig, axs = plt.subplots(1, num_channels, figsize=(fig_width, fig_height), dpi=dpi)

        for i in range(num_channels):
            im = axs[i].imshow(symmetric_filter[i].detach().cpu().numpy())
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

