import torch
from torch import nn
from _tools import mini_tool as tool
from _tools import cnn_frame as cnn


def train(net, num_gpus, batch_size, lr):
    def init_weights(m):
        if type(m) in [nn.Linear, nn.Conv2d]:
            nn.init.normal_(m.weight, std=0.01)

    train_iter, test_iter = tool.load_fashion_mnist(batch_size)
    devices = [tool.try_gpu(i) for i in range(num_gpus)]
    net.apply(init_weights)
    # 多GPU
    net = nn.DataParallel(net, device_ids=devices)
    cross_entropy = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr)
    timer, num_epochs = tool.Timer(), 10
    animator = tool.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        print('epoch {}/{} begin...'.format(epoch + 1, num_epochs))
        net.train()
        timer.start()
        for X, y in train_iter:
            optimizer.zero_grad()
            X, y = X.to(devices[0]), y.to(devices[0])
            # 不用显式地在不同GPU上操作
            loss = cross_entropy(net(X), y)
            loss.backward()
            optimizer.step()
        timer.stop()
        animator.add(epoch + 1, (cnn.evaluate_accuracy_gpu(net, test_iter),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch on {str(devices)}')


if __name__ == '__main__':
    net, devices = cnn.resnet18(10), tool.try_all_gpus()
