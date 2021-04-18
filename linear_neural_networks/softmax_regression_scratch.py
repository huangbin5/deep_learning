import torch
from _tools import mini_tool as tool
from _tools import mlp_frame as mlp


def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition


def model(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])


def optimizer(batch_size):
    return mlp.sgd([W, b], lr, batch_size)


# 因为load_data使用了多线程，因此要放在main函数里面
if __name__ == '__main__':
    # 读取数据集
    batch_size, num_epochs, lr = 256, 5, 0.1
    train_iter, test_iter = tool.load_fashion_mnist(batch_size)
    # 初始化参数
    num_inputs, num_outputs = 784, 10
    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)
    # 训练模型
    mlp.train(model, cross_entropy, optimizer, train_iter, test_iter, num_epochs)
    mlp.predict(model, test_iter)
