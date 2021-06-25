import torch
from torch import nn
from torch.utils import data
from torchvision import transforms
from torchvision import datasets
from matplotlib import pyplot as plt
from _tools.constant import *
from _tools import mini_tool as tool
from _tools import cv_frame as cv


class G_block(nn.Module):
    def __init__(self, out_channels, in_channels=3, kernel_size=4, strides=2, padding=1):
        super(G_block, self).__init__()
        self.conv2d_trans = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, strides, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d_trans(X)))


class D_block(nn.Module):
    def __init__(self, out_channels, in_channels=3, kernel_size=4, strides=2, padding=1, alpha=0.2):
        super(D_block, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(alpha, inplace=True)

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d(X)))


def train(net_D, net_G, data_iter, num_epochs, lr, latent_dim, device=tool.try_gpu()):
    loss = nn.BCEWithLogitsLoss(reduction='sum')
    for w in net_D.parameters():
        nn.init.normal_(w, 0, 0.02)
    for w in net_G.parameters():
        nn.init.normal_(w, 0, 0.02)
    net_D, net_G = net_D.to(device), net_G.to(device)
    trainer_hp = {'lr': lr, 'betas': [0.5, 0.999]}
    optimizer_D = torch.optim.Adam(net_D.parameters(), **trainer_hp)
    optimizer_G = torch.optim.Adam(net_G.parameters(), **trainer_hp)
    animator = tool.Animator(xlabel='epoch', ylabel='loss', xlim=[1, num_epochs], nrows=2, figsize=(5, 5),
                             legend=['discriminator', 'generator'])
    animator.fig.subplots_adjust(hspace=0.3)
    for epoch in range(1, num_epochs + 1):
        print('epoch {}/{} begin...'.format(epoch, num_epochs))
        timer = tool.Timer()
        metric = tool.Accumulator(3)
        for X, _ in data_iter:
            batch_size = X.shape[0]
            Z = torch.normal(0, 1, size=(batch_size, latent_dim, 1, 1))
            X, Z = X.to(device), Z.to(device)
            metric.add(cv.update_D(X, Z, net_D, net_G, loss, optimizer_D),
                       cv.update_G(Z, net_D, net_G, loss, optimizer_G), batch_size)
        # 显示生成效果
        Z = torch.normal(0, 1, size=(21, latent_dim, 1, 1), device=device)
        # (-1, 1)标准化到(0, 1)
        fake_x = net_G(Z).permute(0, 2, 3, 1) / 2 + 0.5
        imgs = torch.cat([torch.cat([fake_x[i * 7 + j].cpu().detach() for j in range(7)], dim=1)
                          for i in range(len(fake_x) // 7)], dim=0)
        animator.axes[1].cla()
        animator.axes[1].imshow(imgs)
        loss_D, loss_G = metric[0] / metric[2], metric[1] / metric[2]
        animator.add(epoch, (loss_D, loss_G))
    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, {metric[2] / timer.stop():.1f} examples/sec on {str(device)}')
    plt.show()


if __name__ == '__main__':
    DATA_HUB['pokemon'] = (DATA_URL + 'pokemon.zip', 'c065c0e2593b8b161a2d7873e42418bf6a21106c')
    data_dir = tool.download_extract('pokemon')
    transformer = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)])
    pokemon = datasets.ImageFolder(data_dir, transform=transformer)
    batch_size = 256
    data_iter = data.DataLoader(pokemon, batch_size=batch_size, shuffle=True, num_workers=4)

    n_G = 64
    # 将一个向量转化为一张图像：(100,) -> (100, 1, 1) -> (3, 64, 64)
    net_G = nn.Sequential(
        # Output: (64 * 8, 4, 4)
        G_block(in_channels=100, out_channels=n_G * 8, strides=1, padding=0),
        # Output: (64 * 4, 8, 8)
        G_block(in_channels=n_G * 8, out_channels=n_G * 4),
        # Output: (64 * 2, 16, 16)
        G_block(in_channels=n_G * 4, out_channels=n_G * 2),
        # Output: (64, 32, 32)
        G_block(in_channels=n_G * 2, out_channels=n_G),
        # Output: (3, 64, 64)
        nn.ConvTranspose2d(in_channels=n_G, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),
        nn.Tanh())
    n_D = 64
    net_D = nn.Sequential(
        # Output: (64, 32, 32)
        D_block(n_D),
        # Output: (64 * 2, 16, 16)
        D_block(in_channels=n_D, out_channels=n_D * 2),
        # Output: (64 * 4, 8, 8)
        D_block(in_channels=n_D * 2, out_channels=n_D * 4),
        # Output: (64 * 8, 4, 4)
        D_block(in_channels=n_D * 4, out_channels=n_D * 8),
        # Output: (1, 1, 1)
        nn.Conv2d(in_channels=n_D * 8, out_channels=1, kernel_size=4, bias=False))
    latent_dim, lr, num_epochs = 100, 0.005, 1
    train(net_D, net_G, data_iter, num_epochs, lr, latent_dim)
