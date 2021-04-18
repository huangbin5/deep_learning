import torch
from torch import nn
from _tools import mini_tool as tool
from _tools import mlp_frame as mlp


class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, is_train=True):
        # python3 中也可以写成 super().__init__()
        super(Net, self).__init__()
        self.train = is_train
        self.num_inputs = num_inputs
        self.linear1 = nn.Linear(num_inputs, num_hiddens1)
        self.linear2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.linear3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.linear1(X.reshape((-1, self.num_inputs))))
        # 只在训练的时候dropout
        if self.train:
            H1 = dropout(H1, p1)
        H2 = self.relu(self.linear2(H1))
        if self.train:
            H2 = dropout(H2, p2)
        out = self.linear3(H2)
        return out


def dropout(X, p):
    assert 0 <= p <= 1
    if p == 0:
        return X
    # 所有神经元dropout
    if p == 1:
        return torch.zeros_like(X)
    mask = (torch.zeros(X.shape).uniform_(0, 1) > p).float()
    # *是元素相乘，@是矩阵相乘
    return mask * X / (1.0 - p)


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


if __name__ == '__main__':
    batch_size, num_epochs, lr = 256, 10, 0.5
    train_iter, test_iter = tool.load_fashion_mnist(batch_size)
    p1, p2 = 0.2, 0.5
    num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
    model = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
    cross_entropy = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    mlp.train(model, cross_entropy, optimizer, train_iter, test_iter, num_epochs)

    # 简明实现
    model = nn.Sequential(nn.Flatten(), nn.Linear(num_inputs, num_hiddens1), nn.ReLU(), nn.Dropout(p1),
                          nn.Linear(num_hiddens1, num_hiddens2), nn.ReLU(), nn.Dropout(p2),
                          nn.Linear(num_hiddens2, num_outputs))
    model.apply(init_weights)
    mlp.train(model, cross_entropy, optimizer, train_iter, test_iter, num_epochs)
