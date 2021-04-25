import time
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils import data
from torchvision import datasets
from torchvision import transforms
import os
import hashlib
import tarfile
import zipfile
import requests
from _tools.constant import *


class Timer:
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()

    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        return sum(self.times) / len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1, figsize=(7, 5)):
        # 画布、子图
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # todo 暂时不清楚多个子图在这里如何绘制
        self.config_axes = lambda: set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()


def synthetic_array(w, b, num_data):
    X = torch.normal(0, 1, (num_data, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def load_array(data_arrays, batch_size, is_train=True):
    # 将features和labels组合成样本
    data_set = data.TensorDataset(*data_arrays)
    return data.DataLoader(data_set, batch_size, shuffle=is_train)


def load_fashion_mnist(batch_size, resize=None):
    # 将[0, 255]内的取值转换为[0.0, 1.0]内的取值
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    # 将多个transforms组合到一起
    trans = transforms.Compose(trans)
    # 读取数据
    mnist_train = datasets.FashionMNIST(root="../_data", train=True, transform=trans, download=True)
    mnist_test = datasets.FashionMNIST(root="../_data", train=False, transform=trans, download=True)
    # 返回迭代器，多线程读取
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4))


def get_fashion_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def download(name, cache_dir=os.path.join('..', '_data')):
    assert name in DATA_HUB, f"{name} does not exist in {DATA_HUB}."
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    # 本地存储文件名
    file_name = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(file_name):
        sha1 = hashlib.sha1()
        with open(file_name, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            # 已有本地文件，直接返回
            return file_name
    print(f'Downloading {file_name} from {url}...')
    response = requests.get(url, stream=True, verify=True)
    with open(file_name, 'wb') as f:
        f.write(response.content)
    return file_name


def download_extract(name, folder=None):
    file_name = download(name)
    base_dir = os.path.dirname(file_name)
    file_dir, ext = os.path.splitext(file_name)
    if ext == '.zip':
        fp = zipfile.ZipFile(file_name, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(file_name, 'r')
    else:
        # 只允许zip和tar压缩文件
        assert False, 'Only zip/tar files can be extracted.'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else file_dir


def download_all():
    for name in DATA_HUB:
        download(name)


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置画图参数"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    # 对于yscale为'log'时要先设置yscale再设置ylim
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    # 显示网格
    axes.grid()


def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(7, 5), axes=None, show=True):
    plt.figure(figsize=figsize)
    axes = axes if axes else plt.gca()

    # 只有一维数据
    def has_one_axis(X):
        return hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list) and not hasattr(X[0], "__len__")

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        assert len(X) == 1
        X = X * len(Y)
    # 清除子图
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    if show:
        plt.show()


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    plt.show()
    return axes


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]
