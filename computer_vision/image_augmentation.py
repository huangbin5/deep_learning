import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
from _tools import mini_tool as tool
from _tools import cv_frame as cv
from _tools import cnn_frame as cnn


def apply(img, aug, num_rows=2, num_cols=4, scale=2):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    tool.show_images(Y, num_rows, num_cols, scale=scale)


def augment_test():
    img = Image.open('../_data/img/cat1.jpg')

    # 翻转
    # apply(img, transforms.RandomHorizontalFlip())
    # apply(img, transforms.RandomVerticalFlip())

    # 裁剪
    # apply(img, transforms.RandomResizedCrop((200, 200), scale=(0.1, 1), ratio=(0.5, 2)))

    # 改变颜色：色调、饱和度、亮度、对比度
    # apply(img, transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0))
    # apply(img, transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.5))
    # apply(img, transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))

    # 组合多种变换方式
    augs = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop((200, 200), scale=(0.1, 1), ratio=(0.5, 2)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)])
    apply(img, augs)

    plt.show()


def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(m.weight)


def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = cv.load_cifar10(True, train_augs, batch_size)
    test_iter = cv.load_cifar10(False, test_augs, batch_size)
    cross_entropy = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    cv.train(net, train_iter, test_iter, cross_entropy, optimizer, 10, tool.try_all_gpus())


if __name__ == '__main__':
    # augment_test()

    train_augs = transforms.Compose([transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor()])
    test_augs = transforms.Compose([transforms.ToTensor()])

    net, batch_size = cnn.resnet18(10, 3), 256
    net.apply(init_weights)
    train_with_data_aug(train_augs, test_augs, net)
