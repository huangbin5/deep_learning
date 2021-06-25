import math
import torch
from torch import nn
from torch.optim import lr_scheduler
from _tools import mini_tool as tool
from _tools import mlp_frame as mlp
from _tools import cnn_frame as cnn
from matplotlib import pyplot as plt


def net_fn():
    class Reshape(nn.Module):
        def forward(self, x):
            return x.view(-1, 1, 28, 28)

    return torch.nn.Sequential(
        Reshape(),
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(), nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
        nn.Linear(120, 84), nn.ReLU(),
        nn.Linear(84, 10))


def train(net, train_iter, test_iter, num_epochs, cross_entropy, optimizer, device, scheduler=None):
    net.to(device)
    animator = tool.Animator(xlabel='epoch', xlim=[0, num_epochs], legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        print('epoch {}/{} begin...'.format(epoch + 1, num_epochs))
        metric = tool.Accumulator(3)
        for i, (X, y) in enumerate(train_iter):
            net.train()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            loss = cross_entropy(y_hat, y)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(loss * X.shape[0], mlp.count_accurate(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter), (train_loss, train_acc, None))
        test_acc = cnn.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))

        # lr scheculer
        if scheduler:
            if scheduler.__module__ == lr_scheduler.__name__:
                scheduler.step()
            else:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = scheduler(epoch)

    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')
    plt.show()


class SquareRootScheduler:
    def __init__(self, lr=0.1):
        self.lr = lr

    def __call__(self, num_update):
        return self.lr * pow(num_update + 1.0, -0.5)


class FactorScheduler:
    def __init__(self, factor=1, stop_factor_lr=1e-7, base_lr=0.1):
        self.factor = factor
        self.stop_factor_lr = stop_factor_lr
        self.base_lr = base_lr

    def __call__(self, num_update):
        self.base_lr = max(self.stop_factor_lr, self.base_lr * self.factor)
        return self.base_lr


class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0, warmup_steps=0, warmup_begin_lr=0):
        self.max_update = max_update
        self.base_lr_orig = base_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (self.base_lr_orig - self.final_lr) * (
                        1 + math.cos(math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr


if __name__ == '__main__':
    batch_size, lr, num_epochs = 256, 0.3, 30
    train_iter, test_iter = tool.load_fashion_mnist(batch_size=batch_size)
    device, net = tool.try_gpu(), net_fn()
    cross_entropy = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    # 自定义调度器
    # scheduler = SquareRootScheduler(lr=0.1)
    # scheduler = FactorScheduler(factor=0.9, stop_factor_lr=1e-2, base_lr=2.0)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30], gamma=0.5)
    scheduler = CosineScheduler(max_update=20, base_lr=0.3, final_lr=0.01)
    train(net, train_iter, test_iter, num_epochs, cross_entropy, optimizer, device, scheduler)
