import torch
import torch.nn as nn
import numpy as np

class Channel(nn.Module):
    def __init__(self, channel_type='awgn'):
        super().__init__()
        self.channel_type = channel_type

    def forward(self, x, snr_db, h_real=None, h_imag=None, b_prob=None, b_stddev=None):
        b, c, height, width = x.shape
        x_flat = x.view(b, -1)
        dim = x_flat.size(1)

        power = torch.mean(x_flat ** 2, dim=1, keepdim=True)
        norm_factor = torch.sqrt(dim / power)
        x_flat = x_flat * norm_factor

        snr_linear = 10 ** (snr_db.view(-1, 1) / 10)
        noise_std = torch.sqrt(1 / snr_linear)

        if self.channel_type == 'awgn':
            noise = torch.randn_like(x_flat) * noise_std
            out = x_flat + noise

        elif self.channel_type == 'slow_fading':
            h = torch.sqrt(torch.tensor(0.5)) * torch.randn_like(x_flat)
            x_flat = x_flat * h
            noise = torch.randn_like(x_flat) * noise_std
            out = x_flat + noise

        elif self.channel_type == 'slow_fading_eq':
            h = torch.sqrt(torch.tensor(0.5)) * torch.randn_like(x_flat)
            h = h.clamp(min=1e-2)  # 避免除以接近0
            noise = torch.randn_like(x_flat) * noise_std
            out = x_flat + noise / h

        elif self.channel_type == 'burst':
            noise = torch.randn_like(x_flat) * noise_std
            b_mask = (torch.rand_like(x_flat) < b_prob.view(-1, 1)).float()
            b_noise = b_mask * torch.randn_like(x_flat) * b_stddev.view(-1, 1)
            out = x_flat + noise + b_noise

        else:
            raise ValueError(f"Unsupported channel type: {self.channel_type}")

        return out.view(b, c, height, width)

