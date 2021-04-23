from torch import nn
from _tools import mini_tool as tool
from _tools import cnn_frame as cnn


# ResNet-18
def resnet_module(input_channels, num_channels, num_residuals, first_module=False):
    blk = []
    for i in range(num_residuals):
        if not first_module and i == 0:
            blk.append(cnn.Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(cnn.Residual(num_channels, num_channels))
    return blk


if __name__ == '__main__':
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b2 = nn.Sequential(*resnet_module(64, 64, 2, first_module=True))
    b3 = nn.Sequential(*resnet_module(64, 128, 2))
    b4 = nn.Sequential(*resnet_module(128, 256, 2))
    b5 = nn.Sequential(*resnet_module(256, 512, 2))
    net = nn.Sequential(b1, b2, b3, b4, b5, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(512, 10))

    import torch
    print(net)
    X = torch.rand(size=(1, 1, 224, 224))
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)

    lr, num_epochs, batch_size = 0.05, 10, 256
    train_iter, test_iter = tool.load_fashion_mnist(batch_size, resize=96)
    cnn.train(net, train_iter, test_iter, num_epochs, lr, tool.try_gpu())
