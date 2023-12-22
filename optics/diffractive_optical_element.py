import torch
import torchvision
from torch import nn, Tensor

from Spectraltask import data_save_path
from constant import MATERIAL_REFRACTIVE_INDEX_FUNCS
from constant import wave_length_list_400_700nm
import numpy


def _phase_to_height_with_material_refractive_idx_func(_phase, _wavelength, _refractive_index_function):
    return _phase / (2 * torch.pi / _wavelength) / (_refractive_index_function(_wavelength) - 1)


def create_Rotationally_radius_mesh(height_map_shape):
    radius = height_map_shape
    diameter = 2 * height_map_shape
    u = torch.linspace(-diameter // 2, diameter // 2 - 1, diameter, dtype=torch.float32)
    v = torch.linspace(-diameter // 2, diameter // 2 - 1, diameter, dtype=torch.float32)
    [x, y] = torch.meshgrid(u, v, indexing='xy')
    radius_distance = torch.sqrt(x ** 2 + y ** 2)
    return radius_distance


class RotationallySymmetricQuantizedHeightMapDOELayer(nn.Module):
    def __init__(self, image_patch=(512, 512), _quantization_level_cnt=4, BASE_PLANE_THICKNESS=2 * 1e-3,
                 quantization_height_base_wavelength=700 * 1e-9, height_tolerance=2e-8, glass_name='SK1300',
                 wave_length=wave_length_list_400_700nm, device='cuda'):
        super(RotationallySymmetricQuantizedHeightMapDOELayer, self).__init__()
        self.height_map_weight = nn.parameter.Parameter(data=torch.ones(image_patch[0], dtype=torch.float32) * 1e-4,
                                                        requires_grad=True)
        self.height_map_shape = nn.Parameter(torch.tensor(image_patch[0]), requires_grad=False)
        self._quantization_level_cnt = _quantization_level_cnt
        # print("[QDO] Quantization levels：", _quantization_level_cnt)
        self.base_plane_thickness = BASE_PLANE_THICKNESS
        self.quantization_height_base_wavelength = quantization_height_base_wavelength
        self.wavelength_to_refractive_index_func = MATERIAL_REFRACTIVE_INDEX_FUNCS[glass_name]
        self.quantization_base_height = torch.tensor(_phase_to_height_with_material_refractive_idx_func(
            _phase=2 * torch.pi,  # 2*pi,可以调节以控制不同的刻蚀高度，如4pi对应2h_max刻的更深
            _wavelength=quantization_height_base_wavelength,
            _refractive_index_function=self.wavelength_to_refractive_index_func),
            dtype=torch.float32)  # Δn*h_max=λ (h_max in paper)
        self.radius_distance = nn.Parameter(create_Rotationally_radius_mesh(self.height_map_shape), requires_grad=False)
        self.height_tolerance = height_tolerance
        self.wavelength = nn.Parameter(torch.from_numpy(wave_length), requires_grad=False)
        self.device = device
        self.aperture = nn.Parameter(circular_aperture(self.height_map_shape),
                                     requires_grad=False)  # set the circular aperture

    def forward(self, Input_field, step, writer):
        preprocess_height = preprocess_height_map(self.height_map_weight, self._quantization_level_cnt,
                                                  self.base_plane_thickness, self.quantization_base_height,
                                                  self.height_map_shape, self.radius_distance)
        base_plane_height_map = add_height_map_noise(preprocess_height, tolerance=self.height_tolerance,
                                                     device=self.device)
        # if step % 3 == 0:
        #     writer.add_image(tag='phase distribution',
        #                      img_tensor=preprocess_height.view(1, Input_field.shape[1], Input_field.shape[2])
        #                                 /torch.max(preprocess_height), global_step=step)
        if step % 10 == 0:
            prim_height = preprocess_height.view(1, Input_field.shape[1], Input_field.shape[2]) * self.aperture.view(1,
                                                                                                                     2 * self.height_map_shape,
                                                                                                                     2 * self.height_map_shape)
            prim_height[self.aperture.view(1, 2 * self.height_map_shape, 2 * self.height_map_shape) == 1] \
                = self.base_plane_thickness - prim_height[
                self.aperture.view(1, 2 * self.height_map_shape, 2 * self.height_map_shape) == 1]
            writer.add_image(tag='phase distribution',
                             img_tensor=prim_height / torch.max(prim_height), global_step=step)
        field_after_element = shift_phase_according_to_height_map(self.wavelength_to_refractive_index_func,
                                                                  self.wavelength, height_map=base_plane_height_map,
                                                                  Input_field=Input_field)
        field_after_element = field_after_element * self.aperture.view(1, 2 * self.height_map_shape,
                                                                       2 * self.height_map_shape, 1)

        return field_after_element


def _quantized_path(_weight, _round_op, _quantization_level_cnt=4, _adaptive=False):
    def _norm_to_0_and_1(_x):
        return (_x + 1) * 0.5  # tf.sigmoid(_x) [0, 1.414]

    def _full_precise_path(_weight):
        return _norm_to_0_and_1(_weight)

    _normed_height_map = _full_precise_path(torch.clamp(_weight, -1, 1))  # clip to [-1, 1]

    '''pytorch is not able to derive this quantization operation like tensorflow does'''
    # quantized_levels = torch.round(_normed_height_map * (_quantization_level_cnt - 1))
    # _quantized_height_map = quantized_levels / (_quantization_level_cnt - 1)

    _quantized_height_map = _normed_height_map
    return _quantized_height_map


def phase_base_plane_height_map(base_plane_thickness, quantization_base_height, x):
    return base_plane_thickness - quantization_base_height * x


def preprocess_height_map(height_map_weight, _quantization_level_cnt, base_plane_thickness,
                          quantization_base_height, height_map_shape, radius_distance):
    # --- With STE
    height_map_weight_1D = phase_base_plane_height_map(base_plane_thickness, quantization_base_height,
                                                       _quantized_path(height_map_weight,
                                                                       torch.round,
                                                                       _quantization_level_cnt))
    radius = height_map_shape
    radius_distance = radius_distance

    Rotationally_Height_map = torch.where((radius_distance >= 0) & (radius_distance <= 1), height_map_weight_1D[0],
                                          0)
    for i in torch.arange(1, radius):
        Rotationally_Height_map += torch.where((radius_distance > i) & (radius_distance <= i + 1),
                                               height_map_weight_1D[i], 0)

    height_map = Rotationally_Height_map.view(1, 2 * radius, 2 * radius, 1)  # 整形为四维

    return height_map


def shift_phase_according_to_height_map(wavelength_to_refractive_index_func, wavelength, height_map, Input_field):
    delta_n = wavelength_to_refractive_index_func(wavelength) - 1
    delta_n = delta_n.reshape((1, 1, 1, -1))
    k = 2. * torch.pi / wavelength
    k = k.reshape((1, 1, 1, -1))  # (1, 1, 1, 31)
    phi = torch.exp(1j * k * delta_n * height_map).type(torch.complex64)
    shifted_field = Input_field * phi
    return shifted_field


def circular_aperture(height_map_shape):
    input_shape = 2 * height_map_shape
    aperture = torch.zeros(input_shape, input_shape)
    u = torch.linspace(-input_shape // 2, input_shape // 2 - 1, input_shape)
    v = u
    [x, y] = torch.meshgrid(u, v, indexing='xy')
    r = torch.sqrt(x ** 2 + y ** 2)
    aperture[r <= input_shape // 2] = 1
    return aperture


def add_height_map_noise(height_map, tolerance=None, device='cuda'):
    if tolerance is not None:
        height_map += torch.Tensor(*height_map.shape).uniform_(-tolerance, tolerance).to(device)
    return height_map


class RotationallySymmetricPhase(nn.Module):
    def __init__(self, sampling_interval=4e-6, sampling_point_number=1024, Order=5):
        super(RotationallySymmetricPhase, self).__init__()
        Radius = sampling_interval * sampling_point_number / 2
        self.coefficient = nn.Parameter(data=torch.zeros(Order, dtype=torch.float32), requires_grad=True)
        self.radius_distance = nn.Parameter(create_Rotationally_radius_mesh(sampling_point_number // 2),
                                            requires_grad=False)
        self.coordinates_map_1D = torch.linspace(0, sampling_point_number // 2 - 1,
                                                 sampling_point_number // 2,
                                                 dtype=torch.float32) * sampling_interval / Radius
        self.coordinates_map_1D = nn.Parameter(self.coordinates_map_1D, requires_grad=False)
        self.Order = Order
        self.sampling_point_number = sampling_point_number
        self.wavelength = nn.Parameter(torch.from_numpy(wave_length_list_400_700nm), requires_grad=False)
        self.aperture = nn.Parameter(circular_aperture(sampling_point_number // 2), requires_grad=False)

    def forward(self, Input_field, step, writer):
        f = torch.zeros_like(self.coordinates_map_1D)
        for power in range(self.Order):
            f += self.coefficient[power] * self.coordinates_map_1D ** (power * 2)
        f = f * torch.max(self.wavelength)
        Rotationally_phase_map = torch.where((self.radius_distance >= 0) & (self.radius_distance <= 1), f[0],
                                             0)
        for i in torch.arange(1, self.sampling_point_number // 2):
            Rotationally_phase_map += torch.where((self.radius_distance > i) & (self.radius_distance <= i + 1),
                                                  f[i], 0)

        Rotationally_phase_map_4D = Rotationally_phase_map.view(1, self.sampling_point_number,
                                                                self.sampling_point_number, 1)
        k = 2. * torch.pi / self.wavelength
        k = k.reshape((1, 1, 1, -1))

        if step % 20 == 0:
            writer.add_image(tag='phase distribution',
                             img_tensor=Rotationally_phase_map.view(1, self.sampling_point_number,
                                                                    self.sampling_point_number) / torch.max(f),
                             global_step=step)
            torchvision.utils.save_image(
                Rotationally_phase_map.view(1, self.sampling_point_number, self.sampling_point_number) / torch.max(f),
                data_save_path + '/Height_map' + '/step' + str(step) + '.jpg')

        phi = torch.exp(1j * k * Rotationally_phase_map_4D).type(torch.complex64)
        output = Input_field * phi
        output_field = output * self.aperture.view(1, self.sampling_point_number, self.sampling_point_number, 1)
        return output_field


