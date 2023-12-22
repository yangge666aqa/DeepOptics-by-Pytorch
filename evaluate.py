import torch
from metrics import psnr_metric


def psnr_eval(dataloader, model, epoch, step, writer):
    # 初始化test_loss 和 correct， 用来统计每次的误差
    PSNR = 0
    size = 0
    # 测试时模型参数不用更新，所以no_gard()
    with torch.no_grad():
        # 加载数据加载器，得到里面的X（图片数据）和y(真实标签）
        for X in dataloader:
            size += 1
            # 将数据转到GPU
            X = X.cuda()
            # 将图片传入到模型当中就，得到预测的值pred
            pred = model(X, step)
            # 计算预测值pred和真实值y的差距
            PSNR += psnr_metric(pred, X).item()

    PSNR /= size

    print('epoch ' + str(epoch) + ' | ' + 'PSNR in validation data : '
          + str(PSNR))

    return PSNR