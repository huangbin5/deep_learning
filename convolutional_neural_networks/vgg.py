from torch import nn
from _tools import mini_tool as tool
from _tools import cnn_frame as cnn


def vgg_block(num_convs, in_channels, out_channels):
    """多个卷积层(+激活层)，最后跟着一个池化层"""
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def vgg(conv_arch):
    conv_blks, in_channels = [], 1
    # 根据conv_arch参数添加卷积块
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    return nn.Sequential(*conv_blks, nn.Flatten(),
                         nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
                         nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
                         nn.Linear(4096, 10))


if __name__ == '__main__':
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    net = vgg(conv_arch)
    small_conv_arch = [(pair[0], pair[1] // 4) for pair in conv_arch]
    net = vgg(small_conv_arch)

    lr, num_epochs, batch_size = 0.05, 10, 128
    train_iter, test_iter = tool.load_fashion_mnist(batch_size, resize=224)
    cnn.train(net, train_iter, test_iter, num_epochs, lr, cnn.try_gpu())
