import math
import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt
from _tools import mini_tool as tool
from _tools import mlp_frame as mlp


def train(train_features, test_features, train_labels, test_labels, num_epochs=400):
    batch_size = min(10, train_features.shape[0])
    train_iter = tool.load_array((train_features, train_labels.reshape(-1, 1)), batch_size)
    test_iter = tool.load_array((test_features, test_labels.reshape(-1, 1)), batch_size, is_train=False)

    # 关闭bias，features第一列是0次幂，相当于是bias
    model = nn.Sequential(nn.Linear(train_features.shape[-1], 1, bias=False))
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    animator = tool.Animator(xlabel='epoch', ylabel='loss', xlim=[1, num_epochs], ylim=[1e-3, 1e2], yscale='log',
                             legend=['train', 'test'])
    for epoch in range(num_epochs):
        mlp.train_epoch(model, mse_loss, optimizer, train_iter)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            print('epoch {}/{} begin...'.format(epoch + 1, num_epochs))
            animator.add(epoch + 1, (mlp.evaluate_loss(model, train_iter, mse_loss),
                                     mlp.evaluate_loss(model, test_iter, mse_loss)))
    print('weight:', model[0].weight.data.numpy())
    plt.show()


"""
for model selection
"""
if __name__ == '__main__':
    # 生成数据
    max_degree, num_train, num_test = 20, 100, 100
    features = np.random.normal(size=(num_train + num_test, 1))
    np.random.shuffle(features)
    # 特征就是x的每个次幂
    features = np.power(features, np.arange(max_degree).reshape(1, -1))
    # x^n/gamma(n) = x^n/(n-1)!
    for i in range(max_degree):
        features[:, i] /= math.gamma(i + 1)
    true_w = np.zeros(max_degree)
    # 生成最高次幂是3次的数据
    true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])
    # (200, 20) * (20,) = (200,)
    labels = np.dot(features, true_w)
    labels += np.random.normal(scale=0.1, size=labels.shape)
    # 转化为tensor类型
    true_w, features, labels = [torch.tensor(x, dtype=torch.float32) for x in [true_w, features, labels]]

    # 选用和生成数据集时同样的3次幂去拟合
    train(features[:num_train, :4], features[num_train:, :4], labels[:num_train], labels[num_train:])
    # 选用1次幂去拟合 -> underfitting
    train(features[:num_train, :2], features[num_train:, :2], labels[:num_test], labels[num_test:])
    # 选用20次幂去拟合 -> overfitting
    train(features[:num_train, :], features[num_train:, :], labels[:num_test], labels[num_test:], num_epochs=1500)
