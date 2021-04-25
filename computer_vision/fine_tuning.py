import os
import torch
from torch.utils import data
from torchvision import datasets
from torchvision import transforms
from torchvision import models
from torch import nn
from _tools.constant import *
from _tools import mini_tool as tool
from _tools import cv_frame as cv


def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5, param_group=True):
    train_iter = data.DataLoader(datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_augs),
                                 batch_size=batch_size, shuffle=True)
    test_iter = data.DataLoader(datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=test_augs),
                                batch_size=batch_size)
    cross_entropy = nn.CrossEntropyLoss(reduction="none")
    if param_group:
        # 用10倍的学习率训练输出层参数
        params_1x = [param for name, param in net.named_parameters() if name not in ["fc.weight", "fc.bias"]]
        optimizer = torch.optim.SGD([{'params': params_1x}, {'params': net.fc.parameters(), 'lr': learning_rate * 10}],
                                    lr=learning_rate, weight_decay=0.001)
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.001)
    cv.train(net, train_iter, test_iter, cross_entropy, optimizer, num_epochs, tool.try_all_gpus())


if __name__ == '__main__':
    DATA_HUB['hotdog'] = (DATA_URL + 'hotdog.zip', 'fba480ffa8aa7e0febbb511d181409f899b9baa5')
    data_dir = tool.download_extract('hotdog')

    train_imgs = datasets.ImageFolder(os.path.join(data_dir, 'train'))
    test_imgs = datasets.ImageFolder(os.path.join(data_dir, 'test'))
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_augs = transforms.Compose(
        [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    test_augs = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])

    finetune_net = models.resnet18(pretrained=True)
    finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
    nn.init.xavier_uniform_(finetune_net.fc.weight)
    train_fine_tuning(finetune_net, 5e-5)
