import random
import torch
from _tools import mini_tool as tool
from _tools import mlp_frame as mlp


# 获取批数据
def data_iter(features, labels, batch_size):
    num_data = len(features)
    indices = list(range(num_data))
    random.shuffle(indices)
    for i in range(0, num_data, batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_data)])
        yield features[batch_indices], labels[batch_indices]


if __name__ == '__main__':
    # 生成数据
    true_w, true_b = torch.tensor([2, -3.4]), 4.2
    features, labels = tool.synthetic_array(true_w, true_b, 1000)

    # 初始化参数
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    batch_size, num_epochs, lr = 10, 3, 0.03

    # 训练模型
    for epoch in range(num_epochs):
        for X, y in data_iter(features, labels, batch_size):
            loss = mlp.squared_loss(mlp.linear_regression(X, w, b), y)
            loss.sum().backward()
            mlp.sgd([w, b], lr, batch_size)
        with torch.no_grad():
            train_loss = mlp.squared_loss(mlp.linear_regression(features, w, b), labels)
            print('epoch {}, loss {:f}'.format(epoch + 1, train_loss.mean()))
