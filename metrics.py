import torch


def psnr_metric(ground_truth, prediction):
    mse = torch.mean(torch.square(ground_truth - prediction))
    psnr = torch.subtract(
        (20.0 * torch.log(torch.tensor(1.0)) / torch.log(torch.tensor(10.0))),
        ((10.0 / torch.log(torch.tensor(10.0))) * torch.log(mse)))
    return psnr
