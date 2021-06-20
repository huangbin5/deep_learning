import torch
from _tools import rnn_frame as rnn
from _tools import mini_tool as tool


def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size
    normal = lambda shape: torch.randn(size=shape, device=device) * 0.01
    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 添加梯度信息
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_rnn_state(batch_size, num_hiddens, device):
    # 初始状态全为0。返回元组，状态变量有可能多个
    return torch.zeros((batch_size, num_hiddens), device=device),


def rnn_layer(inputs, state, params):
    """inputs.shape = (num_steps, batch_size, vocab_size)
       outputs.shape = (num_steps * batch_size, vocab_size)"""
    W_xh, W_hh, b_h, W_hq, b_q = params
    outputs, (H,) = [], state
    for X in inputs:
        H = torch.tanh((X @ W_xh) + (H @ W_hh) + b_h)
        Y = (H @ W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)


if __name__ == '__main__':
    batch_size, num_steps = 32, 35
    train_iter, vocab = rnn.load_data_time_machine(batch_size, num_steps)

    num_hiddens, num_epochs, lr = 512, 500, 1
    net = rnn.RNNModelScratch(len(vocab), num_hiddens, tool.try_gpu(), get_params, init_rnn_state, rnn_layer)
    # train(net, train_iter, vocab, lr, num_epochs, cnn.try_gpu())

    rnn.train(net, train_iter, vocab, lr, num_epochs, tool.try_gpu(), use_random_iter=True)
