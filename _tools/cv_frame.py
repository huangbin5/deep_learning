from torch.utils import data
from torchvision import datasets
from torch import nn
from _tools import mini_tool as tool
from _tools import mlp_frame as mlp
from _tools import cnn_frame as cnn


def resnet18(num_classes, in_channels=1):
    def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(cnn.Residual(in_channels, out_channels, use_1x1conv=True, strides=2))
            else:
                blk.append(cnn.Residual(out_channels, out_channels))
        return nn.Sequential(*blk)

    # 使用更小的卷积核，移除了max pooling层
    net = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU())
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1, 1)))
    net.add_module("fc", nn.Sequential(nn.Flatten(), nn.Linear(512, num_classes)))
    return net


def load_cifar10(is_train, augs, batch_size):
    dataset = datasets.CIFAR10(root="../_data", train=is_train, transform=augs, download=True)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=4)
    return dataloader


def train_batch(net, X, y, cross_entropy, optimizer, devices):
    if isinstance(X, list):
        # Required for BERT Fine-tuning
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    optimizer.zero_grad()
    pred = net(X)
    loss = cross_entropy(pred, y)
    loss.sum().backward()
    optimizer.step()
    train_loss_sum = loss.sum()
    train_acc_sum = mlp.count_accurate(pred, y)
    return train_loss_sum, train_acc_sum


def train(net, train_iter, test_iter, cross_entropy, trainer, num_epochs, devices=tool.try_all_gpus()):
    timer, num_batches = tool.Timer(), len(train_iter)
    animator = tool.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                             legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        print('epoch {}/{} begin...'.format(epoch + 1, num_epochs))
        # (sum of loss, num of accurate, num of examples)
        metric = tool.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            loss, acc = train_batch(net, features, labels, cross_entropy, trainer, devices)
            # todo 后面两个数字一样的？
            metric.add(loss, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (metric[0] / metric[2], metric[1] / metric[3], None))
        test_acc = cnn.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc {metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(devices)}')
