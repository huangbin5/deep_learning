from torch import nn
from _tools import mini_tool as tool
from _tools import cnn_frame as cnn


def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    """由1个自定义大小卷积层紧接着2个1×1卷积层"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())


if __name__ == '__main__':
    net = nn.Sequential(
        nin_block(1, 96, kernel_size=11, strides=4, padding=0), nn.MaxPool2d(3, stride=2),
        nin_block(96, 256, kernel_size=5, strides=1, padding=2), nn.MaxPool2d(3, stride=2),
        nin_block(256, 384, kernel_size=3, strides=1, padding=1), nn.MaxPool2d(3, stride=2), nn.Dropout(0.5),
        # 标签类别数是10
        nin_block(384, 10, kernel_size=3, strides=1, padding=1),
        # 全局平均池化层
        nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

    lr, num_epochs, batch_size = 0.1, 10, 128
    train_iter, test_iter = tool.load_fashion_mnist(batch_size, resize=224)
    cnn.train(net, train_iter, test_iter, num_epochs, lr, tool.try_gpu())
