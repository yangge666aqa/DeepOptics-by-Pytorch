import torch
from constant import SRF_GS3_U3_41S4C_BGR_31_CHANNEL_400_700NM


def simulated_rgb_camera_spectral_response_function(hyper_spectral_image, device):  # Shape=(batch, height, width, 31)
    channel_num = 31
    masked_response_function = SRF_GS3_U3_41S4C_BGR_31_CHANNEL_400_700NM.to(device)  # back (3, 31) as the response value for bgr
    # Red, we use the normalization for all channels
    red_response = hyper_spectral_image * masked_response_function[2].reshape(shape=[1, 1, 1, channel_num])
    # Here, data flow is (1,512,512,31) * (1,1,1,31) = (1,512,512,31)
    red_channel = torch.sum(red_response, dim=-1) / torch.sum(masked_response_function[2])
    # Green
    green_response = hyper_spectral_image * masked_response_function[1].reshape(shape=[1, 1, 1, channel_num])
    green_channel = torch.sum(green_response, dim=-1) / torch.sum(masked_response_function[1])
    # Blue
    blue_response = hyper_spectral_image * masked_response_function[0].reshape(shape=[1, 1, 1, channel_num])
    blue_channel = torch.sum(blue_response, dim=-1) / torch.sum(masked_response_function[0])
    # Stack RGB channels
    rgb_image = torch.stack([red_channel, green_channel, blue_channel], dim=-1)
    # Shape=(batch, height, width, 3)
    return rgb_image

