"""Validation with explicit evaluation mode and device handling."""
import torch
from deepoptics.metrics import psnr_metric


def psnr_eval(dataloader, model, epoch=0, step=0, writer=None):
    """Mean batch PSNR, preserving the original aggregation convention."""
    total, batches = 0.0, 0
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for data in dataloader:
                data = data.to(device=device, dtype=torch.float32)
                total += psnr_metric(model(data, step), data).item()
                batches += 1
    finally:
        model.train(was_training)
    if not batches:
        raise ValueError('Validation dataset is empty.')
    score = total / batches
    print(f'epoch {epoch} | PSNR in validation data: {score}')
    return score
