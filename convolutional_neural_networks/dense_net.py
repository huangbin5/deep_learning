import torch
from torch import nn
from _tools import mini_tool as tool
from _tools import cnn_frame as cnn


def conv_block(input_channels, num_channels):
    # dense block使用了ResNet改进的"批归一化、激活层、卷积层"结构
    return nn.Sequential(nn.BatchNorm2d(input_channels), nn.ReLU(),
                         nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))


class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super().__init__()
        layer = []
        for i in range(num_convs):
            # 由于每次将输出拼接起来，因此经过每个卷积块，通道数就会增加num_channels
            layer.append(conv_block(num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 将它们在通道维度上连接
            X = torch.cat((X, Y), dim=1)
        return X


def transition_block(input_channels, num_channels):
    return nn.Sequential(nn.BatchNorm2d(input_channels), nn.ReLU(),
                         nn.Conv2d(input_channels, num_channels, kernel_size=1),
                         nn.AvgPool2d(kernel_size=2, stride=2))


if __name__ == '__main__':
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    num_channels, growth_rate, num_convs_in_dense_blocks = 64, 32, [4] * 4
    blks = []
    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        blks.append(DenseBlock(num_convs, num_channels, growth_rate))
        # 上一个dense block的输出通道数
        num_channels += num_convs * growth_rate
        # 在dense block之间添加一个transition layer，使通道数量减半
        if i != len(num_convs_in_dense_blocks) - 1:
            blks.append(transition_block(num_channels, num_channels // 2))
            num_channels //= 2
    net = nn.Sequential(b1, *blks, nn.BatchNorm2d(num_channels), nn.ReLU(),
                        nn.AdaptiveMaxPool2d((1, 1)), nn.Flatten(), nn.Linear(num_channels, 10))

    lr, num_epochs, batch_size = 0.1, 10, 256
    train_iter, test_iter = tool.load_fashion_mnist(batch_size, resize=96)
    cnn.train(net, train_iter, test_iter, num_epochs, lr, tool.try_gpu())
