import numpy as np
import torch

def psnr(mse, max_val=1.0):
    return 10 * np.log10((max_val ** 2) / mse)



def crop_size(x: torch.Tensor, y: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    _, _, h1, w1 = x.shape
    _, _, h2, w2 = y.shape
    h = min(h1, h2)
    w = min(w1, w2)
    return x[:, :, :h, :w], y[:, :, :h, :w]
