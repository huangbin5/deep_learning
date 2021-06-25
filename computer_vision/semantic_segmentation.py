import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision import transforms
from _tools.constant import *
from _tools import mini_tool as tool
from _tools import cv_frame as cv


def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1), torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels, kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight


def loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)


def predict(img):
    X = test_iter.dataset.normalize_image(img).unsqueeze(0)
    pred = net(X.to(devices[0])).argmax(dim=1)
    return pred.reshape(pred.shape[1], pred.shape[2])


def label2image(pred):
    colormap = torch.tensor(VOC_COLORMAP, device=devices[0])
    X = pred.long()
    return colormap[X, :]


def sample_test():
    """ 只裁取固定大小的区域进行展示 """
    test_images, test_labels = cv.read_voc_images(False)
    n, imgs = 4, []
    for i in range(n):
        crop_rect = (0, 0, 320, 480)
        X = transforms.F.crop(test_images[i], *crop_rect)
        pred = label2image(predict(X))
        imgs += [X.permute(1, 2, 0), pred.cpu(), transforms.F.crop(test_labels[i], *crop_rect).permute(1, 2, 0)]
    tool.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2)


if __name__ == '__main__':
    pretrained_net = models.resnet18(pretrained=True)
    net = nn.Sequential(*list(pretrained_net.children())[:-2])
    num_classes = len(VOC_CLASSES)
    net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
    net.add_module('transpose_conv',
                   nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, padding=16, stride=32))
    # 初始化转置卷积的参数
    W = bilinear_kernel(num_classes, num_classes, 64)
    net.transpose_conv.weight.data.copy_(W)

    batch_size, crop_size = 32, (320, 480)
    train_iter, test_iter = cv.load_data_voc(batch_size, crop_size)
    num_epochs, lr, wd, devices = 5, 0.001, 1e-3, tool.try_all_gpus()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
    cv.train(net, train_iter, test_iter, loss, optimizer, num_epochs, devices)
