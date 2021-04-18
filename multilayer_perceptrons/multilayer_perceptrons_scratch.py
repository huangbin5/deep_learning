import torch
from torch import nn
from _tools import mini_tool as tool
from _tools import mlp_frame as mlp


def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)


def model(X):
    X = X.reshape((-1, num_inputs))
    # @符号是矩阵乘法运算符
    H = relu(X @ W1 + b1)
    return H @ W2 + b2


if __name__ == '__main__':
    batch_size, num_epochs, lr = 256, 10, 0.1
    train_iter, test_iter = tool.load_fashion_mnist(batch_size)

    num_inputs, num_outputs, num_hiddens = 784, 10, 256
    W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
    W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
    params = [W1, b1, W2, b2]

    cross_entropy = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params, lr=lr)

    mlp.train(model, cross_entropy, optimizer, train_iter, test_iter, num_epochs)
    mlp.predict(model, test_iter)
