import os
import torch
import pandas as pd
from torch import nn
from torch.utils import data
from torchvision import transforms
from torchvision import datasets
from _tools.constant import *
from _tools import mini_tool as tool
from _tools import cv_frame as cv
from _tools import cnn_frame as cnn


def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay):
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), tool.Timer()
    legend = ['train loss', 'train acc']
    if valid_iter is not None:
        legend.append('valid acc')
    animator = tool.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=legend)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        print('epoch {}/{} begin...'.format(epoch + 1, num_epochs))
        net.train()
        metric = tool.Accumulator(3)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            loss, acc = cv.train_batch(net, features, labels, cross_entropy, optimizer, devices)
            metric.add(loss, acc, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (metric[0] / metric[2], metric[1] / metric[2], None))
        if valid_iter is not None:
            valid_acc = cnn.evaluate_accuracy_gpu(net, valid_iter)
            animator.add(epoch + 1, (None, None, valid_acc))
        scheduler.step()
    measures = f'train loss {metric[0] / metric[2]:.3f}, train acc {metric[1] / metric[2]:.3f}'
    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}'
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(devices)}')


def predict(net, test_iter, classes, devices):
    preds = []
    for X, _ in test_iter:
        y_hat = net(X.to(devices[0]))
        preds.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())
    sorted_ids = list(range(1, len(test_ds) + 1))
    sorted_ids.sort(key=lambda x: str(x))
    df = pd.DataFrame({'id': sorted_ids, 'label': preds})
    df['label'] = df['label'].apply(lambda x: classes[x])
    df.to_csv('../_data/cifar10.csv', index=False)


if __name__ == '__main__':
    demo = True
    if demo:
        DATA_HUB['cifar10_tiny'] = (DATA_URL + 'kaggle_cifar10_tiny.zip', '2068874e4b9a9f0fb07ebe0ad2b29754449ccacd')
        data_dir = tool.download_extract('cifar10_tiny')
    else:
        data_dir = '../_data/cifar-10/'
    batch_size, valid_ratio = 32 if demo else 128, 0.1
    cv.reorg_cifar10_data(data_dir, valid_ratio)
    transform_train = transforms.Compose([
        transforms.Resize(40),
        transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(1.0, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

    """
    ImageFolder返回(X, y)的数据形式，它有3个属性
    classes：所有类别的名称(list)
    class_to_idx：类别对应的索引值(map)
    imgs：每个图像路径及其类别的索引(tuple)
    """
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

    cross_entropy = nn.CrossEntropyLoss(reduction="none")
    devices, num_epochs, lr, wd, num_classes = tool.try_all_gpus(), 20, 2e-4, 5e-4, 10
    lr_period, lr_decay, net = 4, 0.9, cnn.resnet18(num_classes, 3)
    train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay)

    # todo 当超参数调好之后，使用全部训练集和验证集数据进行训练
    # train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period, lr_decay)

    predict(net, test_iter, train_valid_ds.classes, devices)
