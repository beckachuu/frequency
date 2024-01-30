import torch
import torch.fft
from matplotlib import pyplot as plt

from utility.path_utils import get_last_path_element


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


    def save_filter_img(self, save_dir):
        fig, axs = plt.subplots(1, self.filter.shape[0])
        for i in range(self.filter.shape[0]):
            im = axs[i].imshow(self.filter[i].detach().cpu().numpy())
            fig.colorbar(im, ax=axs[i])

        title = get_last_path_element(save_dir).split('.')[0]
        plt.title(title)

        plt.savefig(save_dir)
