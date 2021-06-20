import torch
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
from _tools import mini_tool as tool
from _tools import att_frame as att


def f(x):
    return 2 * torch.sin(x) + x ** 0.8


def plot_kernel_reg(y_hat):
    tool.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'], xlim=[0, 5], ylim=[-1, 5], show=False)
    plt.plot(x_train, y_train, 'o', alpha=0.5)
    plt.show()


def average_pooling():
    y_hat = torch.repeat_interleave(y_train.mean(), n_test)
    plot_kernel_reg(y_hat)


def nonparametric_attention_pooling():
    # repeat_interleave函数将每个元素重复n次。如[1, 2] -> [1, 1, 2, 2]
    # X_repeat的每一行都是同一个query输入
    X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
    # x_train是keys，y_train是values，weights中保存着每个query的查询结果对应values的权重
    attention_weights = F.softmax(-(X_repeat - x_train) ** 2 / 2, dim=1)
    # y_hat是values的加权平均值
    y_hat = torch.matmul(attention_weights, y_train)
    plot_kernel_reg(y_hat)
    att.show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0), xlabel='Sorted training inputs',
                      ylabel='Sorted testing inputs')


class NWKernelRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        # queries.shape = (n,) keys/values.shape = (n, n - 1)
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        self.attention_weights = F.softmax(-((queries - keys) * self.w) ** 2 / 2, dim=1)
        # 批量矩阵相乘 (1, n - 1) @ (n - 1, 1)
        return torch.bmm(self.attention_weights.unsqueeze(1), values.unsqueeze(-1)).reshape(-1)


def parametric_attention_pooling():
    X_tile = x_train.repeat((n_train, 1))
    Y_tile = y_train.repeat((n_train, 1))
    # keys/values.shape = (n, n - 1) 训练时除去自身的key-value对
    keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
    values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))

    net = NWKernelRegression()
    mse_loss = nn.MSELoss(reduction='none')
    optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
    animator = tool.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

    for epoch in range(5):
        optimizer.zero_grad()
        loss = mse_loss(net(x_train, keys, values), y_train) / 2
        loss.sum().backward()
        optimizer.step()
        animator.add(epoch + 1, float(loss.sum()))
        print(f'epoch {epoch + 1}, loss {float(loss.sum()):.6f}')
    plt.show()

    keys = x_train.repeat((n_test, 1))
    values = y_train.repeat((n_test, 1))
    y_hat = net(x_test, keys, values).unsqueeze(1).detach()
    plot_kernel_reg(y_hat)
    att.show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0), xlabel='Sorted training inputs',
                      ylabel='Sorted testing inputs')


if __name__ == '__main__':
    n_train = n_test = 50
    x_train, _ = torch.sort(torch.rand(n_train) * 5)
    y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))
    x_test = torch.arange(0, 5, 0.1)
    y_truth = f(x_test)

    # average_pooling()
    # nonparametric_attention_pooling()
    parametric_attention_pooling()
