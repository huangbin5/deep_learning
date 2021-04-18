import torch
from torch import nn
from _tools import mini_tool as tool
from _tools import mlp_frame as mlp


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


if __name__ == '__main__':
    batch_size, num_epochs, lr = 256, 10, 0.1
    train_iter, test_iter = tool.load_fashion_mnist(batch_size)

    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
    model.apply(init_weights)
    cross_entropy = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    mlp.train(model, cross_entropy, optimizer, train_iter, test_iter, num_epochs)
