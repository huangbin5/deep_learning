import torch
from torch import nn
from _tools import mini_tool as tool
from _tools import mlp_frame as mlp


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


if __name__ == '__main__':
    # 读取数据集
    batch_size, num_epochs, lr = 256, 10, 0.1
    train_iter, test_iter = tool.load_fashion_mnist(batch_size)
    # Flatten层先将图像展开为一维
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    # apply方法递归地使每个module执行init_weights操作
    model.apply(init_weights)
    # CrossEntropyLoss将SoftMax取指数和cross entropy取对数操作合并了
    cross_entropy = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # 训练
    mlp.train(model, cross_entropy, optimizer, train_iter, test_iter, num_epochs)
