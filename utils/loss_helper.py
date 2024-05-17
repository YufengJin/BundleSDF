import torch
import torch.nn.functional as F
import torch.nn as nn

def psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr


def log_mse_loss(input, target):
    log_input = torch.log(input + 1)
    log_target = torch.log(target + 1)
    return F.mse_loss(log_input, log_target)

def l2_loss(input, target, mask=None, *args, **kwargs):
    if mask is not None:
        assert mask.shape == input.shape == target.shape
        return F.mse_loss(input[mask], target[mask], *args, **kwargs)
    else:
        assert input.shape == target.shape
        return F.mse_loss(input, target, *args, **kwargs)

def l1_loss(input, target, mask=None, *args, **kwargs):
    if mask is not None:
        assert mask.shape == input.shape == target.shape
        return F.l1_loss(input[mask], target[mask], *args, **kwargs)
    else:
        assert input.shape == target.shape
        return F.l1_loss(input, target, *args, **kwargs)

