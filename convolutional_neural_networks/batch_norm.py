import torch
from torch import nn
from _tools import mini_tool as tool
from _tools import cnn_frame as cnn


def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 是训练模式还是预测模式
    if not torch.is_grad_enabled():
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 全连接层
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 在通道维度计算均值和方差，保持X的形状以便广播
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动均值和方差(训练模式下不断更新计算，预测模式下使用)
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    # 伸缩和偏移
    Y = gamma * X_hat + beta
    return Y, moving_mean.data, moving_var.data


class BatchNorm(nn.Module):
    # num_features：全连接层的特征数 or 卷积层的通道数
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 伸缩因子=1，偏移=0
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 初始移动均值=0，方差=1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 更新移动均值和方差
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean, self.moving_var, eps=1e-5, momentum=0.9)
        return Y


if __name__ == '__main__':
    net = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(), nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(), nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(), nn.Linear(16 * 4 * 4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
        nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
        nn.Linear(84, 10))
    lr, num_epochs, batch_size = 1.0, 10, 256
    train_iter, test_iter = tool.load_fashion_mnist(batch_size)
    cnn.train(net, train_iter, test_iter, num_epochs, lr, cnn.try_gpu())

    # 简明实现
    net = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(), nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(), nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(), nn.Linear(256, 120), nn.BatchNorm1d(120),
        nn.Sigmoid(), nn.Linear(120, 84), nn.BatchNorm1d(84),
        nn.Sigmoid(), nn.Linear(84, 10))
    cnn.train(net, train_iter, test_iter, num_epochs, lr, cnn.try_gpu())
