import importlib
import math
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from deepoptics.data.dataset_loader import ICVL_512_MAT_Dataset_iter, read_cube
from deepoptics.evaluate import psnr_eval
from deepoptics.metrics import psnr_metric


def test_imports_do_not_create_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for module in ['deepoptics.config', 'deepoptics.train', 'deepoptics.optics.camera']:
        importlib.import_module(module)
    assert list(tmp_path.iterdir()) == []


def test_help_without_training():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, '-m', 'deepoptics', '--help'], cwd=root,
                            capture_output=True, text=True)
    assert result.returncode == 0
    assert '--data-root' in result.stdout


def test_mat_reader(tmp_path):
    cube = np.arange(31 * 3 * 4, dtype=np.uint16).reshape(31, 3, 4)
    path = tmp_path / 'sample.mat'
    with h5py.File(path, 'w') as handle:
        handle['rad'] = cube
    result = read_cube(path)
    assert result.shape == (3, 4, 31)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, cube.transpose(1, 2, 0) / 4095.0, rtol=1e-6)


def test_dataset_errors_and_filter(tmp_path):
    with pytest.raises(ValueError, match='No MAT'):
        ICVL_512_MAT_Dataset_iter(tmp_path)
    with pytest.raises(FileNotFoundError):
        ICVL_512_MAT_Dataset_iter(tmp_path / 'missing')
    (tmp_path / 'ignore.txt').write_text('not a cube')
    with h5py.File(tmp_path / 'cube.mat', 'w') as handle:
        handle['rad'] = np.zeros((31, 512, 512), dtype=np.uint16)
    patches = list(ICVL_512_MAT_Dataset_iter(tmp_path, verbose=False))
    assert len(patches) == 10
    assert all(p.shape == (512, 512, 31) for p in patches)


def test_psnr_and_evaluation_mode():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

        def forward(self, x, step):
            assert not self.training
            return x + 0.1 + self.weight

    model = Model()
    assert psnr_eval([torch.zeros(2, 3)], model) == pytest.approx(20)
    assert model.training
    with pytest.raises(ValueError, match='empty'):
        psnr_eval([], model)
    assert model.training
    assert math.isinf(psnr_metric(torch.zeros(1), torch.zeros(1)).item())


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_camera_forward_backward(device, tmp_path):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable')
    from deepoptics.config import controlled_doe_args, controlled_propagation_args, controlled_network_args
    from deepoptics.optics.camera import Camera
    from deepoptics.losses import ssim_loss
    from deepoptics.optics.sensor_srfs import simulated_rgb_camera_spectral_response_function
    from torch.utils.tensorboard import SummaryWriter
    torch.manual_seed(7)
    model = Camera(wave_resolution=(32, 32), device=device,
                   doe_args=dict(controlled_doe_args, image_patch=(16, 16), height_tolerance=None),
                   propagation_args=controlled_propagation_args,
                   network_args=dict(controlled_network_args, depth=2, filter_root=4, network_input=(16, 16)))
    model = model.to(device).done()
    data = torch.rand(2, 16, 16, 31, device=device)
    for folder in ('MTF', 'PSF'):
        (tmp_path / folder).mkdir()
    model.writer = SummaryWriter(str(tmp_path))
    model.output_dir = tmp_path
    result = model(data, 0)
    assert result.shape == data.shape
    assert torch.isfinite(result).all()
    predicted_rgb = simulated_rgb_camera_spectral_response_function(result, device).permute(0, 3, 1, 2)
    target_rgb = simulated_rgb_camera_spectral_response_function(data, device).permute(0, 3, 1, 2)
    objective = (result - data).abs().mean() + 0.01 * ssim_loss(predicted_rgb, target_rgb)
    objective.backward()
    model.writer.close()
    assert (tmp_path / 'PSF/step0.jpg').is_file()
    assert (tmp_path / 'MTF/step0.png').is_file()
    grad = model.doe_layer.height_map_weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert grad.abs().sum() > 0
    assert model.net.final_conv.weight.grad.abs().sum() > 0
