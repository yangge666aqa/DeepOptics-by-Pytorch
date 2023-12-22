import torch
from util import pytorch_ssim


ssim = pytorch_ssim.SSIM(window_size=11)


def ssim_loss(ground_truth, prediction):
    return 1 - ssim(ground_truth, prediction)


def log_loss(ground_truth, prediction):
    loss = torch.square(torch.log(ground_truth + 1) - torch.log(prediction + 1))
    return loss


LOSS_FUNCTION_FILTER = {
    "mse": torch.nn.MSELoss(),
    "mae": torch.nn.L1Loss(),
    "ssim": ssim_loss,
    "log": log_loss
}