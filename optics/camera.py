import torch
import torchvision.utils
from torch import nn
from Spectraltask import wave_length_list
from optics.diffractive_optical_element import RotationallySymmetricQuantizedHeightMapDOELayer
import numpy as np
from optics.propagation import Propagation
from optics.propagation import transpose_fft, transpose_ifft
from optics.sensor_srfs import simulated_rgb_camera_spectral_response_function
from net.res_unet import Res_Unet
import torchvision.utils as vutils
import time
from Spectraltask import writer
from Spectraltask import controlled_camera_args, controlled_doe_args, controlled_propagation_args, controlled_network_args
import numpy as np
import matplotlib.pyplot as plt
from Spectraltask import data_save_path
import os
import PIL

class Camera(nn.Module):
    def __init__(self, input_channel_num=31, should_use_planar_incidence=False,
                 wave_resolution=(1024, 1024), up_sampling=2, sampling_interval=4e-6, distance=1, device='cuda',
                 doe=RotationallySymmetricQuantizedHeightMapDOELayer, wave_lengths=wave_length_list):
        super(Camera, self).__init__()
        self.flag_use_planar_incidence = should_use_planar_incidence
        self.wave_resolution = wave_resolution
        self.input_channel_num = input_channel_num
        self.wave_length = wave_lengths
        self.sampling_interval = sampling_interval
        self.physical_size = (sampling_interval * self.wave_resolution[0], sampling_interval * self.wave_resolution[1])
        self.image_patch = (self.wave_resolution[0] // up_sampling, self.wave_resolution[1] // up_sampling)
        self.simulated_incidence = None
        self.distance = distance
        self.device = device
        self.doe_layer = doe(**controlled_doe_args).to(device)
        # self.doe_layer = doe().to(device)
        self.propagation = Propagation(**controlled_propagation_args).build_Propagated_kernel(
            input_shape=(*wave_resolution, input_channel_num))
        self.net = Res_Unet(controlled_network_args)
        self.network_input_size = controlled_network_args['network_input']
        # 初始化权重
        self.net.apply(init_weights)

    def done(self):
        """
        Compile the camera. The Camera objection must be called done() before it is called as a `nn.Module` object.

        Returns: Camera object done.
        """
        if self.flag_use_planar_incidence:  # 平面波入射
            self.simulated_incidence = planar_light(self.wave_resolution, self.input_channel_num).to(self.device)
        else:  # 点光源
            self.simulated_incidence = point_source(wave_resolution=self.wave_resolution,
                                                    wave_length_list=self.wave_length,
                                                    physical_size=self.physical_size,
                                                    distance=self.distance).to(self.device)

        return self  # 返回实例本身

    def forward(self, x, step):
        field_before_doe = self.simulated_incidence
        field_after_doe = self.doe_layer(field_before_doe, step, writer)
        field_before_sensor = self.propagation(field_after_doe)
        psf = get_intensities(field_before_sensor)

        psf = psf / torch.sum(psf, dim=(1, 2))
        if (x.shape[1], x.shape[2]) == (psf.shape[1], psf.shape[2]):
            pass
        else:
            x = transpose_resize_image_as(x, psf.shape).to(torch.float32)

        sensor_image_per_wavelength = get_sensor_image_per_wavelength(x, psf, self.device)
        sensor_image = simulated_rgb_camera_spectral_response_function(sensor_image_per_wavelength, self.device)
        # sensor_image = (sensor_image - torch.min(sensor_image)) / (torch.max(sensor_image) - torch.min(sensor_image))
        sensor_image = transpose_resize_image_as(sensor_image, (x.shape[0], *self.network_input_size, x.shape[3]))
        y = self.net(sensor_image)
        if step <= 500:
            if step % 10 == 0:
                writer.add_image(tag='sensor image',
                                 img_tensor=vutils.make_grid(sensor_image[0:sensor_image.shape[0], :, :, :].permute(0, 3, 1, 2)),
                                 global_step=step)
                psf1 = get_psf_for_full_spectrum(psf, (x.shape[0], *(1024, 1024), x.shape[3]))
                # writer.add_image(tag='psf_for_full_spectrum',
                #                  img_tensor=vutils.make_grid(psf1), global_step=step)
                plot_PSF(psf1, step)
                plot_MTF(get_MTF(psf), samlping_interval=4e-6, step=step)  # plot mtf
        else:
            if step % 100 == 0:
                writer.add_image(tag='sensor image',
                                 img_tensor=vutils.make_grid(sensor_image[0:4, :, :, :].permute(0, 3, 1, 2)),
                                 global_step=step)

                psf1 = get_psf_for_full_spectrum(psf, (x.shape[0], *(256, 256), x.shape[3]))
                # writer.add_image(tag='psf_for_full_spectrum',
                #                  img_tensor=vutils.make_grid(psf1), global_step=step)
                plot_PSF(psf1, step)
                plot_MTF(get_MTF(psf), samlping_interval=4e-6, step=step)
        return y.permute(0, 2, 3, 1)


def init_weights(m):
    if type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


def planar_light(wave_resolution, input_channel_num):
    # 返回 shape=(batch_size, wave_resolution, wave_resolution, input_channel_num) 的矩阵
    return torch.ones((wave_resolution[0], wave_resolution[1], input_channel_num), dtype=torch.float32)


def point_source(wave_resolution=(1024, 1024), wave_length_list=None, physical_size=None, point_location=(0, 0),
                 distance=1):
    assert wave_length_list is not None or physical_size is not None, "self.wave_length or self.physical_size can't be None"

    wave_res_n, wave_res_m = wave_resolution
    [y, x] = np.mgrid[-wave_res_n // 2:wave_res_n // 2, -wave_res_m // 2:wave_res_m // 2].astype(np.float64)
    x = x / wave_res_n * physical_size[0]  # -L/2 - L/2
    y = y / wave_res_m * physical_size[1]
    x0, y0 = point_location
    squared_sum = ((x - x0) ** 2 + (y - y0) ** 2 + distance ** 2).reshape(1, *wave_resolution, 1)
    k = (2 * np.pi / wave_length_list).reshape([1, 1, 1, -1])
    phase = k * np.sqrt(squared_sum)  # (1, 1, 1, 31) * (1, 1024, 1024, 1) = (1, 1024, 1024, 31) 这是numpy的一个好trick;
    spherical_wavefront = np.exp(1j * phase, dtype=np.complex64)

    return torch.from_numpy(spherical_wavefront)  # (1, 1024, 1024, 31)


def get_intensities(input_field):
    return torch.square(torch.abs(input_field))


def transpose_resize_image_as(image, size):
    transposed_image = image.permute(0, 3, 1, 2)
    interpolate_tranposed_image = nn.functional.interpolate(transposed_image, size=(size[1], size[2]), mode='bilinear')
    inverse_transpose_image = interpolate_tranposed_image.permute(0, 2, 3, 1)
    return inverse_transpose_image


def get_sensor_image_per_wavelength(x, psf, device):
    _, h, w, _ = psf.shape
    # psf = torch.zeros_like(psf)
    # psf[:, h//2-1:h//2+1, w//2-1:w//2+1, :] = 24903224/3
    spectrum_of_x = transpose_fft(x)  # / torch.sqrt(torch.tensor(h * w))
    psf_shift = ifft_shift_2d_tf(psf, device)
    otf = transpose_fft(psf_shift)  # / torch.sqrt(torch.tensor(h * w))
    sensor_image_per_wavelength = torch.real(
        transpose_ifft(otf * spectrum_of_x))  # 取出实部

    return sensor_image_per_wavelength


def get_MTF(psf, down_sampling=1):
    with torch.no_grad():
        psf1 = psf[:, 0:-1:down_sampling, 0:-1:down_sampling, :] * down_sampling**2
        _, h, w, _ = psf.shape
        otf = transpose_fft(psf1)
        mtf = abs(otf)
    return mtf[:, 0, 0:w//2-1, :]


# 上下左右颠倒
def ifft_shift_2d_tf(a_tensor, device):
    input_shape = a_tensor.shape
    new_tensor = a_tensor
    for axis in range(1, 3):
        n = input_shape[axis]
        split = n - (n + 1) // 2
        temp_list = torch.cat((torch.arange(split, n), torch.arange(split))).to(device)
        new_tensor = torch.index_select(new_tensor, dim=axis, index=temp_list)
    return new_tensor


def get_psf_for_full_spectrum(psf, output_shape):
    # max = torch.max(torch.max(psf[:, :, :, 0:-1:4], dim=1).values, dim=1)
    # min = torch.min(torch.min(psf[:, :, :, 0:-1:4], dim=1).values, dim=1)
    max = torch.max(torch.max(psf[:, :, :, :], dim=1).values, dim=1)
    min = torch.min(torch.min(psf[:, :, :, :], dim=1).values, dim=1)
    # psf1 = (psf[:, :, :, 0:-1:4] - min.values) / (max.values - min.values)
    psf1 = (psf[:, :, :, :] - min.values) / (max.values - min.values)
    psf1 = transpose_resize_image_as(psf1, output_shape)
    return psf1.permute(3, 0, 1, 2)


def plot_MTF(MTF, samlping_interval=4e-6, step=0):
    with torch.no_grad():
        max_frequency = 33
        MTF = np.squeeze(MTF.cpu().detach().numpy())
        figure1 = plt.figure()
        ax1 = figure1.add_subplot(111)
        w, c = MTF.shape
        from matplotlib import cm
        viridis = cm.get_cmap('gist_rainbow', 31)
        colors = [viridis(i) for i in np.linspace(0, 1, c)]
        x = np.linspace(0, 1/(samlping_interval*1e3), w)
        f_max = int(max_frequency//(1/(samlping_interval*1e3)/w))
        for i in range(c-1, -1, -1):
            ax1.plot(x[0:f_max], MTF[0:f_max, i], label=str(int(wave_length_list[i]*1e9))+'nm')
        for i1, j in enumerate(ax1.lines):
            j.set_color(colors[i1])
            ax1.legend(loc="upper right", prop={'size': 6}, bbox_to_anchor=(1, 1), ncol=3)

        plt.axis([0, max_frequency, 0, 1])
        writer.add_figure(tag='MTF for full spectrum', figure=figure1, global_step=step)
        figure1.savefig(data_save_path + '/MTF' + '/step'+str(step))


def plot_PSF(psf, step):
    with torch.no_grad():
        torchvision.utils.save_image(vutils.make_grid(psf[:,:, 256:-256, 256:-256]), data_save_path + '/PSF' + '/step'+str(step)+'.jpg')
        # PSF = vutils.make_grid(psf).permute(1, 2, 0).cpu().detach().numpy()
        # plt.imsave(data_save_path + '/PSF' + '/step'+str(step)+'.jpg', PSF, cmap='gray')


# plt.plot(np.linspace(400,700,31), original_pixel.cpu().detach().numpy().T, label='background truth')
# plt.plot(np.linspace(400,700,31), restruct_pixel.cpu().detach().numpy().T, label='reconstruct image')
# plt.legend(loc="upper left", prop={'size': 10}, bbox_to_anchor=(0, 1), ncol=1)
# plt.xlabel('wavelength/nm')
# plt.ylabel('pixel value')
# plt.savefig(data_save_path + '/MTF' + '/step'+str(step))
# plt.show()
