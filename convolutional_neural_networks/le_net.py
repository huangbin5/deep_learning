import torch
from torch import nn
from _tools import mini_tool as tool
from _tools import cnn_frame as cnn


class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)


if __name__ == '__main__':
    net = nn.Sequential(
        Reshape(), nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(), nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
        nn.Linear(120, 84), nn.Sigmoid(),
        nn.Linear(84, 10))

    batch_size, num_epochs, lr = 256, 10, 0.9
    train_iter, test_iter = tool.load_fashion_mnist(batch_size=batch_size)
    cnn.train(net, train_iter, test_iter, num_epochs, lr, cnn.try_gpu())

'''
loss 0.479, train acc 0.818, test acc 0.813
5231.1 examples/sec on cpu
'''
