import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
from torchvision import datasets
from torchvision import models
from _tools.constant import *
from _tools import mini_tool as tool
from _tools import cv_frame as cv


def get_net(devices):
    finetune_net = nn.Sequential()
    finetune_net.features = models.resnet34(pretrained=True)
    # 修改输出层
    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256), nn.ReLU(), nn.Linear(256, 120))
    finetune_net = finetune_net.to(devices[0])
    # 固定预训练的参数
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net


def evaluate_loss(data_iter, net, devices):
    l_sum, n = 0.0, 0
    for features, labels in data_iter:
        features, labels = features.to(devices[0]), labels.to(devices[0])
        outputs = net(features)
        loss = cross_entropy(outputs, labels)
        l_sum += loss.sum()
        n += labels.numel()
    return l_sum / n


def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay):
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    optimizer = torch.optim.SGD((param for param in net.parameters() if param.requires_grad),
                                lr=lr, momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), tool.Timer()
    legend = ['train loss']
    if valid_iter is not None:
        legend.append('valid loss')
    animator = tool.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=legend)
    for epoch in range(num_epochs):
        print('epoch {}/{} begin...'.format(epoch + 1, num_epochs))
        metric = tool.Accumulator(2)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            features, labels = features.to(devices[0]), labels.to(devices[0])
            optimizer.zero_grad()
            output = net(features)
            loss = cross_entropy(output, labels).sum()
            loss.backward()
            optimizer.step()
            metric.add(loss, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (metric[0] / metric[1], None))
        measures = f'train loss {metric[0] / metric[1]:.3f}'
        if valid_iter is not None:
            valid_loss = evaluate_loss(valid_iter, net, devices)
            animator.add(epoch + 1, (None, valid_loss.detach()))
        scheduler.step()
    if valid_iter is not None:
        measures += f', valid loss {valid_loss:.3f}'
    print(measures + f'\n{metric[1] * num_epochs / timer.sum():.1f} examples/sec on {str(devices)}')


def predict(net, test_iter, classes, devices):
    preds = []
    for X, _ in test_iter:
        output = F.softmax(net(X.to(devices[0])), dim=0)
        preds.extend(output.cpu().detach().numpy())
    ids = sorted(os.listdir(os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')))
    with open('../_data/dog_breed.csv', 'w') as f:
        f.write('id,' + ','.join(classes) + '\n')
        for i, output in zip(ids, preds):
            f.write(i.split('.')[0] + ',' + ','.join([str(num) for num in output]) + '\n')


if __name__ == '__main__':
    demo = True
    if demo:
        DATA_HUB['dog_tiny'] = (DATA_URL + 'kaggle_dog_tiny.zip', '0cb91d09b814ecdc07b50f31f8dcad3e81d6a86d')
        data_dir = tool.download_extract('dog_tiny')
    else:
        data_dir = os.path.join('..', 'data', 'dog-breed-identification')
    batch_size, valid_ratio = 32 if demo else 128, 0.1
    cv.reorg_dog_data(data_dir, valid_ratio)
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_ds, train_valid_ds = [
        datasets.ImageFolder(os.path.join(data_dir, 'train_valid_test', folder), transform=transform_train)
        for folder in ['train', 'train_valid']]
    valid_ds, test_ds = [
        datasets.ImageFolder(os.path.join(data_dir, 'train_valid_test', folder), transform=transform_test)
        for folder in ['valid', 'test']]
    train_iter, train_valid_iter = [
        data.DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
        for dataset in (train_ds, train_valid_ds)]
    valid_iter = data.DataLoader(valid_ds, batch_size, shuffle=False, drop_last=True)
    test_iter = data.DataLoader(test_ds, batch_size, shuffle=False, drop_last=False)

    cross_entropy = nn.CrossEntropyLoss(reduction='none')
    devices, num_epochs, lr, wd = tool.try_all_gpus(), 10, 1e-4, 1e-4
    lr_period, lr_decay, net = 2, 0.9, get_net(devices)
    train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay)

    # todo 当超参数调好之后，使用全部训练集和验证集数据进行训练
    # train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period, lr_decay)

    predict(net, test_iter, train_valid_ds.classes, devices)
