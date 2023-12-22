import torch
from optics.camera import Camera,transpose_resize_image_as
from Spectraltask import controlled_camera_args, train_batch_size, validation_batch, dataset_name, total_epoch
from torchsummary import summary
from loss import LOSS_FUNCTION_FILTER
import torch.optim.lr_scheduler as lr_scheduler
from evaluate import psnr_eval
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from Spectraltask import data_save_path, writer
from loss import ssim_loss
from optics.sensor_srfs import simulated_rgb_camera_spectral_response_function


def prepare_image_dataset(datasets_name):
    from util.data.dataset_loader import DATASET_PATH, ICVL_512_MAT_Dataset_iter
    root_dir = DATASET_PATH[datasets_name]
    train_dir = root_dir + "/train"
    val_dir = root_dir + "/validation"

    train_datasets = ICVL_512_MAT_Dataset_iter(train_dir, verbose=False)
    validation_datasets = ICVL_512_MAT_Dataset_iter(val_dir, verbose=False)

    from torch.utils.data import DataLoader

    train_loader = DataLoader(dataset=train_datasets,
                              batch_size=train_batch_size, num_workers=0)
    validation_loader = DataLoader(dataset=validation_datasets,
                                   batch_size=validation_batch, num_workers=0)
    return train_loader, validation_loader


def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():  #@save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


device = try_gpu()
torch.set_printoptions(precision=8)
train_loader, validation_loader = prepare_image_dataset(dataset_name)
model = Camera(**controlled_camera_args).to('cuda')
model.done()
loss = LOSS_FUNCTION_FILTER['mae']

# for name, param in model.named_parameters():
#     print(name, param.shape)
trainer_for_DOE = torch.optim.Adam(model.doe_layer.parameters(), lr=0.01)
trainer_for_network = torch.optim.Adam(model.net.parameters(), lr=0.001, weight_decay=1e-5)
scheduler1 = lr_scheduler.ExponentialLR(trainer_for_DOE, gamma=0.8)
scheduler2 = lr_scheduler.ExponentialLR(trainer_for_network, gamma=0.8)


print('The diffractive network dataflow with ' + str(train_batch_size) + ' batchsize')

# print(summary(model, (1024, 1024, 31), batch_size=train_batch_size))
step = 0
PSNR = psnr_eval(validation_loader, model, 0, step, writer)
writer.add_scalar(tag='PSNR', scalar_value=PSNR, global_step=0)
for epoch in range(total_epoch):

    for data in train_loader:

        trainer_for_network.zero_grad()
        trainer_for_DOE.zero_grad()

        data = data.to(device)
        output = model(data, step)
        original_image = simulated_rgb_camera_spectral_response_function(data, device)
        restruct_image = simulated_rgb_camera_spectral_response_function(output, device)
        ssim = ssim_loss(restruct_image.permute(0, 3, 1, 2), original_image.permute(0, 3, 1, 2).type(torch.float32))
        l = loss(output, data) + 1e-2 * ssim
        l.backward()
        trainer_for_DOE.step()
        trainer_for_network .step()

        writer.add_scalar(tag='loss', scalar_value=l, global_step=step)
        with torch.no_grad():
            if step % 100 == 0:
                # original_image = simulated_rgb_camera_spectral_response_function(data, device)
                writer.add_image(tag='original image',
                                 img_tensor=vutils.make_grid(original_image.permute(0, 3, 1, 2)), global_step=step)
            if step % 100 == 0:
                # restruct_image = simulated_rgb_camera_spectral_response_function(output, device)
                writer.add_image(tag='reconstruct image',
                                 img_tensor=vutils.make_grid(restruct_image.permute(0, 3, 1, 2)), global_step=step)
            if step % 10 == 0:
                print('epoch ' + str(epoch) + ' | ' + 'step ' + str(step) + ' | ' + 'loss : ' + str(l.item()))

        # if step % 100 == 0:
        #     # writer.export_scalars_to_json(data_save_path+'/json')
        step += 1
    torch.save(model.state_dict(), data_save_path + '/pt' + str(step))
    PSNR = psnr_eval(validation_loader, model, epoch + 1, step, writer)
    writer.add_scalar(tag='PSNR', scalar_value=PSNR, global_step=epoch + 1)
    scheduler1.step()
    scheduler2.step()

writer.export_scalars_to_json(data_save_path + '/summary.json')

