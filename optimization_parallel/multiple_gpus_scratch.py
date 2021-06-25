import torch
from torch import nn
from torch.nn import functional as F
from _tools import mini_tool as tool
from _tools import mlp_frame as mlp
from _tools import cnn_frame as cnn

scale = 0.01
W1 = torch.randn(size=(20, 1, 3, 3)) * scale
b1 = torch.zeros(20)
W2 = torch.randn(size=(50, 20, 5, 5)) * scale
b2 = torch.zeros(50)
W3 = torch.randn(size=(800, 128)) * scale
b3 = torch.zeros(128)
W4 = torch.randn(size=(128, 10)) * scale
b4 = torch.zeros(10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]


def lenet(X, params):
    h1_conv = F.conv2d(input=X, weight=params[0], bias=params[1])
    h1_activation = F.relu(h1_conv)
    h1 = F.avg_pool2d(input=h1_activation, kernel_size=(2, 2), stride=(2, 2))
    h2_conv = F.conv2d(input=h1, weight=params[2], bias=params[3])
    h2_activation = F.relu(h2_conv)
    h2 = F.avg_pool2d(input=h2_activation, kernel_size=(2, 2), stride=(2, 2))
    h2 = h2.reshape(h2.shape[0], -1)
    h3_linear = torch.mm(h2, params[4]) + params[5]
    h3 = F.relu(h3_linear)
    y_hat = torch.mm(h3, params[6]) + params[7]
    return y_hat


def get_params(params, device):
    new_params = [p.to(device) for p in params]
    for p in new_params:
        p.requires_grad_()
    return new_params


def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].to(data[0].device)
    for i in range(1, len(data)):
        data[i] = data[0].to(data[i].device)


def train_batch(X, y, device_params, devices, lr):
    # 分布式存储数据
    X_shards, y_shards = tool.split_batch(X, y, devices)
    # 损失分别计算
    losses = [cross_entropy(lenet(X_shard, device_W), y_shard).sum()
              for X_shard, y_shard, device_W in zip(X_shards, y_shards, device_params)]
    for loss in losses:
        loss.backward()
    # 聚合计算梯度
    with torch.no_grad():
        for i in range(len(device_params[0])):
            allreduce([device_params[c][i].grad for c in range(len(devices))])
    for param in device_params:
        mlp.sgd(param, lr, X.shape[0])


def train(num_gpus, batch_size, lr, num_epochs):
    train_iter, test_iter = tool.load_fashion_mnist(batch_size)
    devices = [tool.try_gpu(i) for i in range(num_gpus)]
    device_params = [get_params(params, d) for d in devices]
    animator = tool.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    timer = tool.Timer()
    for epoch in range(num_epochs):
        print('epoch {}/{} begin...'.format(epoch + 1, num_epochs))
        timer.start()
        for X, y in train_iter:
            train_batch(X, y, device_params, devices, lr)
            # 同步各个设备的数据
            torch.cuda.synchronize()
        timer.stop()
        animator.add(epoch + 1,
                     (cnn.evaluate_accuracy_gpu(lambda x: lenet(x, device_params[0]), test_iter, devices[0]),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch on {str(devices)}')


if __name__ == '__main__':
    cross_entropy = nn.CrossEntropyLoss(reduction='none')
    train(num_gpus=2, batch_size=256, lr=0.2, num_epochs=10)
