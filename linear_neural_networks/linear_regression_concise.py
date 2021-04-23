import torch
from torch import nn
from _tools import mini_tool as tool

if __name__ == '__main__':
    # 生成并加载数据集
    true_w, true_b = torch.tensor([2, -3.4]), 4.2
    features, labels = tool.synthetic_array(true_w, true_b, 1000)
    batch_size, num_epochs, lr = 10, 3, 0.03
    data_iter = tool.load_array((features, labels), batch_size)

    # 模型、策略、算法
    model = nn.Sequential(nn.Linear(2, 1))
    model[0].weight.data.normal_(0, 0.01)
    model[0].bias.data.fill_(0)
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # 训练
    for epoch in range(num_epochs):
        for X, y in data_iter:
            optimizer.zero_grad()
            loss = mse_loss(model(X), y)
            loss.backward()
            optimizer.step()
        loss = mse_loss(model(features), labels)
        print('epoch {}, loss {:f}'.format(epoch + 1, loss))
