import pandas as pd
import torch
from torch import nn
import numpy as np
from _tools.constant import *
from _tools import mini_tool as tool


# 使用对数均方根误差作为
def log_rmse(y_hat, y):
    # 将小于1的数变成1
    clipped_preds = torch.clamp(y_hat, 1, float('inf'))
    rmse = torch.sqrt(mse_loss(torch.log(clipped_preds), torch.log(y)))
    return rmse.item()


def train(model, train_features, train_labels, valid_features, valid_labels, num_epochs, lr, weight_decay, batch_size):
    train_loss, valid_loss = [], []
    train_iter = tool.load_array((train_features, train_labels), batch_size)
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            loss = mse_loss(model(X), y)
            loss.backward()
            optimizer.step()
        train_loss.append(log_rmse(model(train_features), train_labels))
        if valid_labels is not None:
            valid_loss.append(log_rmse(model(valid_features), valid_labels))
    return train_loss, valid_loss


def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        # todo 如果不整除的话遍历完k组数据是不是还有剩余？？？
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs, lr, weight_decay, batch_size):
    train_loss_sum, valid_loss_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        model = nn.Sequential(nn.Linear(X_train.shape[1], 1))
        train_loss, valid_loss = train(model, *data, num_epochs, lr, weight_decay, batch_size)
        train_loss_sum += train_loss[-1]
        valid_loss_sum += valid_loss[-1]
        if i == 0:
            tool.plot(list(range(1, num_epochs + 1)), [train_loss, valid_loss], xlabel='epoch', ylabel='rmse',
                      xlim=[1, num_epochs], legend=['train', 'valid'], yscale='log')
        print(f'fold {i + 1}, train log rmse {float(train_loss[-1]):f}, valid log rmse {float(valid_loss[-1]):f}')
    return train_loss_sum / k, valid_loss_sum / k


def train_and_pred(train_features, train_labels, test_features, test_data, num_epochs, lr, weight_decay, batch_size):
    model = nn.Sequential(nn.Linear(train_features.shape[1], 1))
    # 用所有训练集进行训练
    train_loss, _ = train(model, train_features, train_labels, None, None, num_epochs, lr, weight_decay, batch_size)
    tool.plot(np.arange(1, num_epochs + 1), [train_loss], xlabel='epoch', ylabel='log rmse',
              xlim=[1, num_epochs], yscale='log')
    print(f'train log rmse {float(train_loss[-1]):f}')
    # 将网络应用于测试集
    preds = model(test_features).detach().numpy()
    # 将其重新格式化以导出到Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(-1))
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('../_data/house_prices.csv', index=False)


if __name__ == '__main__':
    DATA_HUB['kaggle_house_train'] = (DATA_URL + 'kaggle_house_pred_train.csv',
                                      '585e9cc93e70b39160e7921475f9bcd7d31219ce')
    DATA_HUB['kaggle_house_test'] = (DATA_URL + 'kaggle_house_pred_test.csv',
                                     'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')
    # 读取数据集
    train_data = pd.read_csv(tool.download('kaggle_house_train'))
    test_data = pd.read_csv(tool.download('kaggle_house_test'))
    # 将训练集和测试集中的第一列id删除，并除去训练集中最后一列标签，剩下的就是特征
    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

    # 数据预处理
    numeric_index = all_features.dtypes[all_features.dtypes != 'object'].index
    # 将数值型特征数据标准化(训练集+测试集一起)，缺失值填充为平均值即0
    all_features[numeric_index] = all_features[numeric_index].apply(lambda x: (x - x.mean()) / (x.std()))
    all_features[numeric_index] = all_features[numeric_index].fillna(0)
    # 使用one-hot编码处理离散值类型。dummy_na表示是否将空值也当做一个新的特征
    all_features = pd.get_dummies(all_features, dummy_na=True)

    num_train = train_data.shape[0]
    # 使用values属性从pandas格式中提取出numpy格式，并转化为tensor
    train_features = torch.tensor(all_features[:num_train].values, dtype=torch.float32)
    test_features = torch.tensor(all_features[num_train:].values, dtype=torch.float32)
    train_labels = torch.tensor(train_data['SalePrice'].values.reshape(-1, 1), dtype=torch.float32)

    mse_loss = nn.MSELoss()
    batch_size, num_epochs, lr, weight_decay, k = 64, 100, 5, 0, 5
    train_loss, valid_loss = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
    print(f'{k}-fold validation: avg train log rmse: {float(train_loss):f}, avg valid log rmse: {float(valid_loss):f}')

    train_and_pred(train_features, train_labels, test_features, test_data, num_epochs, lr, weight_decay, batch_size)
