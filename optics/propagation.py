import torch
from torch import nn
from constant import wave_length_list_400_700nm
import numpy
import torch.fft as FFT

class Propagation(nn.Module):
    '''定义传播器'''
    def __init__(self, sensor_distance=0.05, discretization_size=4e-6, wave_lengths=wave_length_list_400_700nm, method='Fresnel', device='cuda', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distance = sensor_distance  # 0.05 m
        self.wave_lengths = wave_lengths
        self.discretization_size = discretization_size  # 像素尺寸 pixel size 4e-6 m
        self.m_padding = None
        self.n_padding = None
        self.outside_sensor_boolean_mask = None
        self.method = method
        self.device = device
        self.H = None

    def build_Propagated_kernel(self, input_shape):
        m_original, n_original, channel_num = input_shape
        # set the value of padding , Here we simply set the quarter of the original size
        m_padding = m_original // 4
        n_padding = n_original // 4
        self.m_padding = m_padding
        self.n_padding = n_padding
        # compute the full size after padding
        m_full = m_original + 2 * m_padding
        n_full = n_original + 2 * n_padding
        m_lins = torch.linspace(-m_full//2, m_full/2-1, m_full)
        n_lins = torch.linspace(-n_full // 2, n_full / 2 - 1, n_full)
        [x, y] = torch.meshgrid(m_lins, n_lins, indexing='xy')  # pixel_shape
        fx = x / (self.discretization_size * m_full)  # (-M/2L, M/2L)
        fy = y / (self.discretization_size * n_full)
        '''搬移频谱，和正常的搬移方式不同，但也可行'''
        fx, fy = FFT.ifftshift(fx), FFT.ifftshift(fy)
        fx = fx.view(1, m_full, n_full, 1)
        fy = fy.view(1, m_full, n_full, 1)
        if isinstance(self.wave_lengths, numpy.ndarray):
            self.wave_lengths = torch.from_numpy(self.wave_lengths)
        self.wave_lengths = self.wave_lengths.view(1, 1, 1, channel_num)

        k = 2. * torch.pi / self.wave_lengths

        if self.method == 'Fresnel':
            self.H = torch.exp(1j * k * self.distance - 1j * torch.pi * self.distance * self.wave_lengths * (fx**2+fy**2))\
                .type(torch.complex64).to(self.device)
        elif self.method == 'Angular Spectrum':
            U_m, V_m = \
                fx.expand(1, m_full, m_full, channel_num).clone(), fy.expand(1, m_full, m_full, channel_num).clone()
            ind = (fx ** 2 + fy ** 2) >= (1 / self.wave_lengths) ** 2
            U_m[ind], V_m[ind] = 0, 0
            self.H = torch.exp(1j * 2 * torch.pi * self.distance * torch.sqrt((1 / self.wave_lengths) ** 2 - U_m ** 2 -
                     V_m ** 2), ).type(torch.complex64).to(self.device)

        return self

    def forward(self, input_field):  # input_field_size = (1, 1024, 1024, 31)
        # padding input field
        # Here we pad the 1/4 aperture size of original size
        _, h, w, _ = input_field.shape
        padded_input_field = transpose_zero_pad(input_field, self.m_padding, self.n_padding)
        sprectrum_of_input_field = transpose_fft(padded_input_field)
        field_before_sensor = transpose_ifft(sprectrum_of_input_field * self.H)
        return field_before_sensor[:, self.m_padding:-self.m_padding, self.n_padding:-self.n_padding, :] # delete padding region


def transpose_zero_pad(input_field, x_padding, y_padding):  # x <--> w <--> m, y <--> h <--> n
    transverse_input_field = input_field.permute(0, 3, 1, 2)  # (1, 31, 1024, 1024)
    padded_input_field = nn.functional.pad(transverse_input_field,
                                           (int(x_padding), int(x_padding), int(y_padding), int(y_padding)),
                                           mode="constant", value=0)
    inverse_transverse_input_field = padded_input_field.permute(0, 2, 3, 1)
    return inverse_transverse_input_field


def transpose_fft(input_field):
    transpose_input_field = input_field.permute(0, 3, 1, 2)
    FFT_input_dield = FFT.fftn(transpose_input_field, dim=(2, 3))
    inverse_transverse_input_field = FFT_input_dield.permute(0, 2, 3, 1)
    return inverse_transverse_input_field

def transpose_ifft(input_field):
    transpose_input_field = input_field.permute(0, 3, 1, 2)
    iFFT_input_dield = FFT.ifftn(transpose_input_field, dim=(2, 3))
    inverse_transverse_input_field = iFFT_input_dield.permute(0, 2, 3, 1)
    return inverse_transverse_input_field
