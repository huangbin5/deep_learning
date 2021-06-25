import torch
from torch import nn
from matplotlib import pyplot as plt
from _tools import mini_tool as tool
from _tools import cv_frame as cv


def train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G, latent_dim, data):
    for w in net_D.parameters():
        nn.init.normal_(w, 0, 0.02)
    for w in net_G.parameters():
        nn.init.normal_(w, 0, 0.02)
    cross_entropy = nn.BCEWithLogitsLoss(reduction='sum')
    optimizer_D = torch.optim.Adam(net_D.parameters(), lr=lr_D)
    optimizer_G = torch.optim.Adam(net_G.parameters(), lr=lr_G)
    animator = tool.Animator(xlabel='epoch', ylabel='loss', xlim=[1, num_epochs], nrows=2, figsize=(5, 5),
                             legend=['discriminator', 'generator'])
    animator.fig.subplots_adjust(hspace=0.3)
    for epoch in range(num_epochs):
        print('epoch {}/{} begin...'.format(epoch + 1, num_epochs))
        timer = tool.Timer()
        metric = tool.Accumulator(3)
        for (X,) in data_iter:
            batch_size = X.shape[0]
            Z = torch.normal(0, 1, size=(batch_size, latent_dim))
            metric.add(cv.update_D(X, Z, net_D, net_G, cross_entropy, optimizer_D),
                       cv.update_G(Z, net_D, net_G, cross_entropy, optimizer_G), batch_size)
        # 显示生成效果
        Z = torch.normal(0, 1, size=(100, latent_dim))
        fake_X = net_G(Z).detach().numpy()
        animator.axes[1].cla()
        animator.axes[1].scatter(data[:, 0], data[:, 1])
        animator.axes[1].scatter(fake_X[:, 0], fake_X[:, 1])
        animator.axes[1].legend(['real', 'generated'])
        loss_D, loss_G = metric[0] / metric[2], metric[1] / metric[2]
        animator.add(epoch + 1, (loss_D, loss_G))
    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, {metric[2] / timer.stop():.1f} examples/sec')
    plt.show()


if __name__ == '__main__':
    X = torch.normal(0.0, 1, (1000, 2))
    A = torch.tensor([[1, 2], [-0.1, 0.5]])
    b = torch.tensor([1, 2])
    data = torch.matmul(X, A) + b
    batch_size = 8
    data_iter = tool.load_array((data,), batch_size)

    net_G = nn.Sequential(nn.Linear(2, 2))
    net_D = nn.Sequential(nn.Linear(2, 5), nn.Tanh(), nn.Linear(5, 3), nn.Tanh(), nn.Linear(3, 1))
    lr_D, lr_G, latent_dim, num_epochs = 0.05, 0.005, 2, 200
    train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G, latent_dim, data[:100].detach().numpy())
