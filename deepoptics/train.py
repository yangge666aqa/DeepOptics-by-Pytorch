"""Training CLI. Importing this module never starts an experiment."""
import argparse
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description='Train the diffractive hyperspectral camera.')
    parser.add_argument('--data-root', type=Path, default=Path('datasets/ICVL'))
    parser.add_argument('--output-dir', type=Path, default=Path('runs'))
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--validation-batch-size', type=int, default=4)
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    return parser


def train(args):
    import json
    import random
    from datetime import datetime
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    import torchvision.utils as vutils
    from deepoptics.config import controlled_camera_args
    from deepoptics.data.dataset_loader import ICVL_512_MAT_Dataset_iter
    from deepoptics.evaluate import psnr_eval
    from deepoptics.losses import LOSS_FUNCTION_FILTER, ssim_loss
    from deepoptics.optics.camera import Camera
    from deepoptics.optics.sensor_srfs import simulated_rgb_camera_spectral_response_function

    if min(args.epochs, args.batch_size, args.validation_batch_size) < 1:
        raise ValueError('Epochs and batch sizes must be positive.')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                          if args.device == 'auto' else args.device)
    train_data = ICVL_512_MAT_Dataset_iter(args.data_root / 'train', verbose=False)
    val_data = ICVL_512_MAT_Dataset_iter(args.data_root / 'validation', verbose=False, shuffle=False)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.validation_batch_size, num_workers=0)
    run_dir = args.output_dir / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    for folder in ('MTF', 'PSF', 'checkpoints'):
        (run_dir / folder).mkdir(parents=True, exist_ok=True)
    metadata = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    metadata['resolved_device'] = str(device)
    (run_dir / 'config.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    with SummaryWriter(str(run_dir)) as writer:
        camera_args = dict(controlled_camera_args, device=str(device))
        model = Camera(**camera_args, writer=writer, output_dir=run_dir).to(device).done()
        doe_optimizer = torch.optim.Adam(model.doe_layer.parameters(), lr=0.01)
        net_optimizer = torch.optim.Adam(model.net.parameters(), lr=0.001, weight_decay=1e-5)
        schedulers = [torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.8)
                      for opt in (doe_optimizer, net_optimizer)]
        step = 0
        writer.add_scalar('PSNR', psnr_eval(val_loader, model, 0, step), 0)
        for epoch in range(args.epochs):
            model.train()
            for data in train_loader:
                data = data.to(device=device, dtype=torch.float32)
                net_optimizer.zero_grad()
                doe_optimizer.zero_grad()
                output = model(data, step)
                original_rgb = simulated_rgb_camera_spectral_response_function(data, device)
                predicted_rgb = simulated_rgb_camera_spectral_response_function(output, device)
                objective = LOSS_FUNCTION_FILTER['mae'](output, data) + 1e-2 * ssim_loss(
                    predicted_rgb.permute(0, 3, 1, 2), original_rgb.permute(0, 3, 1, 2))
                objective.backward()
                doe_optimizer.step()
                net_optimizer.step()
                writer.add_scalar('loss', objective.item(), step)
                if step % 100 == 0:
                    writer.add_image('original image', vutils.make_grid(original_rgb.detach().permute(0, 3, 1, 2)), step)
                    writer.add_image('reconstruct image', vutils.make_grid(predicted_rgb.detach().permute(0, 3, 1, 2)), step)
                if step % 10 == 0:
                    print(f'epoch {epoch} | step {step} | loss {objective.item():.6f}')
                step += 1
            torch.save(model.state_dict(), run_dir / 'checkpoints' / f'epoch-{epoch + 1:03d}.pt')
            writer.add_scalar('PSNR', psnr_eval(val_loader, model, epoch + 1, step), epoch + 1)
            for scheduler in schedulers:
                scheduler.step()
    print(f'Experiment saved to {run_dir.resolve()}')


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        train(args)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
