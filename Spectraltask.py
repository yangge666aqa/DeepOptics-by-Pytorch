import time
from tensorboardX import SummaryWriter
from constant import wave_length_list_400_700nm as wave_length_list
from torch import nn
import os

time = time.time()
data_save_path = './runs/'+str(time)
writer = SummaryWriter('./runs/'+str(time))
os.mkdir(data_save_path + '/MTF')
os.mkdir(data_save_path + '/PSF')

up_sampling = 2
image_patch_size = 512
sampling_interval = 4e-6
dataset_name = "ICVL512-MAT"
train_batch_size = 4
total_epoch = 30
validation_batch = 4
wave_resolution = (image_patch_size * up_sampling, image_patch_size * up_sampling)


controlled_camera_args = {'input_channel_num': 31, "wave_resolution": wave_resolution,
                          "up_sampling": up_sampling , "sampling_interval": sampling_interval, "distance": 1, "device": 'cuda',
                          'wave_lengths': wave_length_list}

controlled_doe_args = {'image_patch': (image_patch_size, image_patch_size), '_quantization_level_cnt': 8, 'BASE_PLANE_THICKNESS': 2 * 1e-3,
                       'quantization_height_base_wavelength': 700 * 1e-9, 'height_tolerance': 2e-8,
                       'glass_name': 'SK1300',
                       'wave_length': wave_length_list, 'device': 'cuda'}

controlled_propagation_args = {'sensor_distance': 0.05, 'discretization_size': sampling_interval,
                               'wave_lengths': wave_length_list, 'method': 'Angular Spectrum',
                               'device': 'cuda'}

controlled_network_args = {'depth': 7, 'net_num': 1, 'filter_root': 32, 'image_channels': 3,
                           'remove_first_long_connection': False, 'network_input': (image_patch_size, image_patch_size),
                           'final_activation': nn.Sigmoid, 'final_output_channels': 31, 'tensorboard_writer': writer}
