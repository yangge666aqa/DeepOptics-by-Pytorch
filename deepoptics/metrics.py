"""Metrics for normalized hyperspectral intensities."""
import torch


def psnr_metric(ground_truth, prediction):
    mse = torch.mean(torch.square(ground_truth - prediction))
    return -10.0 * torch.log10(mse)
