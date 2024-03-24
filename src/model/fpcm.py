import torch
import torch.nn as nn
import torch.fft

class FPCM(nn.Module):
    def __init__(self, dim, batches, epoch_lim=40):
        super().__init__()
        self.projection = nn.Conv1d(dim, dim, kernel_size=1, groups=dim)

        self.batches = batches
        self.batch = 0

        self.epoch_lim = epoch_lim
        self.epoch = 1

    def forward(self, x):
        if self.training:
            self.batch += 1;
            if self.batch > self.batches:
                self.batch = 0
                self.epoch += 1

        b, c, h, w = x.shape
        x = x.view(b, c, h*w)
        device = x.device

        Freq = torch.fft.fft(torch.fft.fft(x.permute(0,2,1), dim=-1), dim=-2)

        b, patches, c = Freq.shape
        cutoff_freq = patches // 2 - (patches // 2 - patches // 8) * (self.epoch / self.epoch_lim)
        lowpass_filter_l = torch.exp(-0.5 * torch.square(torch.linspace(0, patches // 2 - 1, patches // 2).unsqueeze(1).repeat(1,c) / (cutoff_freq))).view(1, patches // 2, c)
        lowpass_filter_r = torch.flip(torch.exp(-0.5 * torch.square(torch.linspace(1, patches // 2 , patches // 2).unsqueeze(1).repeat(1,c) / (cutoff_freq))).view(1, patches // 2, c), [1])
        lowpass_filter = torch.concat((lowpass_filter_l, lowpass_filter_r), dim=1).to(device)

        low_Freq = Freq * lowpass_filter
        lowFreq_feature = torch.fft.ifft(torch.fft.ifft(low_Freq, dim=-2), dim=-1).real

        weights = 0.5 * torch.sigmoid(self.projection(x).permute(0,2,1).mean(dim=1)).unsqueeze(dim=1) + 0.5
        out = weights * lowFreq_feature + (1 - weights) * (x.permute(0,2,1) - lowFreq_feature)

        # shape restoration
        out = out.permute(0,2,1)
        out = out.view(b, c, h, w)

        return out

