from torch import nn
from _tools import mini_tool as tool
from _tools import cnn_frame as cnn

if __name__ == '__main__':
    net = nn.Sequential(
        nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Flatten(), nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(p=0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
        nn.Linear(4096, 10))

    batch_size, num_epochs, lr = 128, 10, 0.01
    train_iter, test_iter = tool.load_fashion_mnist(batch_size, resize=224)
    cnn.train(net, train_iter, test_iter, num_epochs, lr, tool.try_gpu())
