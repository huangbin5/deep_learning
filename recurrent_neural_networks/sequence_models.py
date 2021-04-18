import torch
from torch import nn
from _tools import mini_tool as tool
from _tools import mlp_frame as mlp


# 一个简单的多层感知机
def get_net():
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)

    net = nn.Sequential(nn.Linear(4, 10), nn.ReLU(), nn.Linear(10, 1))
    net.apply(init_weights)
    return net


def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        print(f'epoch {epoch + 1}, loss: {mlp.evaluate_loss(net, train_iter, loss):f}')


T = 1000
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))  # (1000,)
# tool.plot(time, x, 'time', 'x', xlim=[1, 1000], figsize=(6, 3))

tau = 4
features = torch.zeros((T - tau, tau))  # (996, 4) 996个长度为4的特征
for i in range(tau):
    features[:, i] = x[i:T - tau + i]
labels = x[tau:].reshape((-1, 1))  # (996, 1) 每个特征后一个时刻的值作为标签
batch_size, n_train = 16, 600
train_iter = tool.load_array((features[:n_train], labels[:n_train]), batch_size, is_train=True)

# 训练
loss = nn.MSELoss()
net = get_net()
train(net, train_iter, loss, 5, 0.01)

# 单步预测
onestep_preds = net(features)
# tool.plot([time, time[tau:]], [x.detach().numpy(), onestep_preds.detach().numpy()], 'time', 'x',
#           legend=['data', '1-step preds'], xlim=[1, 1000], figsize=(6, 3))

# k步预测
multistep_preds = torch.zeros(T)
multistep_preds[:n_train + tau] = x[:n_train + tau]
for i in range(n_train + tau, T):
    multistep_preds[i] = net(multistep_preds[i - tau:i].reshape((1, -1)))
# tool.plot([time, time[tau:], time[n_train + tau:]],
#           [x.detach().numpy(), onestep_preds.detach().numpy(), multistep_preds[n_train + tau:].detach().numpy()],
#           'time', 'x', legend=['data', '1-step preds', 'multistep preds'], xlim=[1, 1000], figsize=(6, 3))

# 分别1、4、16、64步预测 todo 没太看懂
max_steps, steps = 64, (1, 4, 16, 64)
features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
for i in range(tau):
    features[:, i] = x[i:i + T - tau - max_steps + 1]
for i in range(tau, tau + max_steps):
    features[:, i] = net(features[:, i - tau:i]).reshape(-1)
tool.plot([time[tau + i - 1:T - max_steps + i] for i in steps],
          [features[:, (tau + i - 1)].detach().numpy() for i in steps], 'time', 'x',
          legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000], figsize=(6, 3))
