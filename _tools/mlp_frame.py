import torch
from matplotlib import pyplot as plt
from _tools import mini_tool as tool


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
