import torch
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
from _tools import mlp_frame as mlp
from _tools import mini_tool as tool


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


# 2维×2维 -> 2维
def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


# 3维×3维 -> 2维
def corr2d_nin(X, K):
    return sum(corr2d(x, k) for x, k in zip(X, K))


# 3维×4维 -> 3维
def corr2d_nin_nout(X, K):
    return torch.stack([corr2d_nin(X, k) for k in K], 0)


# 1×1卷积，也可以用上面的方法计算
def corr2d_1x1(X, K):
    (c_i, h, w), c_o = X.shape, K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))


def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i:i + p_h, j:j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i:i + p_h, j:j + p_w].mean()
    return Y


def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, torch.nn.Module):
        # 进入evaluation模式，不使用dropout和batchnorm
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    # (num of accurate)
    metric = tool.Accumulator(2)
    for X, y in data_iter:
        if isinstance(X, list):
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        metric.add(mlp.count_accurate(net(X), y), y.numel())
    return metric[0] / metric[1]


def train(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    cross_entropy = nn.CrossEntropyLoss()
    animator = tool.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = tool.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # (sum of loss, sum of accurate, num of examples)
        metric = tool.Accumulator(3)
        # 设置为训练模式
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            loss = cross_entropy(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(loss * X.shape[0], mlp.count_accurate(y_hat, y), X.shape[0])
            timer.stop()
            train_loss, train_acc = metric[0] / metric[2], metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (train_loss, train_acc, None))
                print(f'finished {i + 1}/{num_batches} iteration of {epoch + 1}/{num_epochs} epochs')
        print('-------------------------------------------')
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_loss:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(device)}')
    plt.show()


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            # 当需要通过残差块改变通道数时，使用1×1卷积核改变输入通道数，以便和输出相加
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
