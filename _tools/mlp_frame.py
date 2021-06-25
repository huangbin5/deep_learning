import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
from _tools import mini_tool as tool
from _tools.constant import *


def linear_regression(X, w, b):
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    # 以下的操作不会被记录到下次梯度的计算中
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


# y是0-9的数字，而不是one-hot编码
def count_accurate(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(model, data_iter):
    if isinstance(model, torch.nn.Module):
        # 进入evaluation模式，不使用dropout和batchnorm
        model.eval()
    # (num of accuracy, num of examples)
    accum = tool.Accumulator(2)
    for X, y in data_iter:
        accum.add(count_accurate(model(X), y), y.numel())
    return accum[0] / accum[1]


def evaluate_loss(model, data_iter, mseLoss):
    # (sum of losses, num of examples)
    accum = tool.Accumulator(2)
    for X, y in data_iter:
        y_hat = model(X)
        loss = mseLoss(y_hat, y.reshape(y_hat.shape))
        accum.add(loss.sum(), loss.numel())
    return accum[0] / accum[1]


def train_epoch(model, loss_func, optimizer, train_iter):
    # 设置为训练模式
    if isinstance(model, torch.nn.Module):
        model.train()
    # (training loss, num of accuracy, num of examples)
    accum = tool.Accumulator(3)
    for X, y in train_iter:
        y_hat = model(X)
        loss = loss_func(y_hat, y)
        if isinstance(optimizer, torch.optim.Optimizer):
            # PyTorch内置的优化器
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            accum.add(float(loss) * len(y), count_accurate(y_hat, y), y.numel())
        else:
            # 用户自定义优化器
            loss.sum().backward()
            optimizer(X.shape[0])
            accum.add(float(loss.sum()), count_accurate(y_hat, y), y.numel())
    # (training loss, accuracy)
    return accum[0] / accum[2], accum[1] / accum[2]


def train(model, cross_entropy, optimizer, train_iter, test_iter, num_epochs):
    animator = tool.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                             legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        print('epoch {}/{} begin...'.format(epoch + 1, num_epochs))
        train_result = train_epoch(model, cross_entropy, optimizer, train_iter)
        test_result = evaluate_accuracy(model, test_iter)
        animator.add(epoch + 1, train_result + (test_result,))
    plt.show()


def predict(model, test_iter, n=6):
    for X, y in test_iter:
        break
    trues, preds = tool.get_fashion_labels(y), tool.get_fashion_labels(model(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    tool.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])


def train_2d(trainer, steps=20, f_grad=None):
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad)
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print(f'epoch {i + 1}, x1: {float(x1):f}, x2: {float(x2):f}')
    return results


def show_trace_2d(f, results):
    plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = torch.meshgrid(torch.arange(-5.5, 1.0, 0.1), torch.arange(-3.0, 1.0, 0.1))
    plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


def get_optim_data(batch_size=10, n=1500):
    DATA_HUB['airfoil'] = (DATA_URL + 'airfoil_self_noise.dat', '76e5be1548fd8222e5074cf0faae75edff8cf93f')
    data = np.genfromtxt(tool.download('airfoil'), dtype=np.float32, delimiter='\t')
    data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0))
    data_iter = tool.load_array((data[:n, :-1], data[:n, -1]), batch_size, is_train=True)
    return data_iter, data.shape[1] - 1


def train_optim(trainer_fn, states, hyperparams, data_iter, feature_dim, num_epochs=2):
    w = torch.normal(mean=0.0, std=0.01, size=(feature_dim, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    net = lambda X: linear_regression(X, w, b)
    animator = tool.Animator(xlabel='epoch', ylabel='loss', xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, tool.Timer()
    for epoch in range(num_epochs):
        print('epoch {}/{} begin...'.format(epoch + 1, num_epochs))
        for X, y in data_iter:
            loss = squared_loss(net(X), y).mean()
            loss.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n / X.shape[0] / len(data_iter), (evaluate_loss(net, data_iter, squared_loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
    plt.show()
    return timer.cumsum(), animator.Y[0]


def train_optim_concise(trainer_fn, hyperparams, data_iter, num_epochs=4):
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.normal_(m.weight, std=0.01)

    net = nn.Sequential(nn.Linear(5, 1))
    net.apply(init_weights)
    mse_loss = nn.MSELoss()
    optimizer = trainer_fn(net.parameters(), **hyperparams)
    animator = tool.Animator(xlabel='epoch', ylabel='loss', xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, tool.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            optimizer.zero_grad()
            out = net(X)
            y = y.reshape(out.shape)
            loss = mse_loss(out, y) / 2
            loss.backward()
            optimizer.step()
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n / X.shape[0] / len(data_iter), (evaluate_loss(net, data_iter, mse_loss) / 2,))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
