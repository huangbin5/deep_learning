import torch
from torch import nn
from torch.nn import functional as F


# 自定义块
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        # 这里使用ReLU的函数版本，它在nn.functional模块中定义
        return self.out(F.relu(self.hidden(X)))


# 自定义顺序块
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for block in args:
            # _modules是OrderedDict类型的，会记录元素插入的顺序
            self._modules[block] = block

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X


# 允许在块中执行自定义代码
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的常数参数，其在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 自定义隐层，该层的参数是常数参数，不会被更新
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层，相当于两个全连接层共享参数
        X = self.linear(X)
        # 自定义操作
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()


# 可以进行各种各样的组合
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))
