import torch
from _tools import mini_tool as tool
from _tools import rnn_frame as rnn


def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size
    normal = lambda shape: torch.randn(size=shape, device=device) * 0.01
    triple = lambda: (normal((num_inputs, num_hiddens)), normal((num_hiddens, num_hiddens)),
                      torch.zeros(num_hiddens, device=device))

    W_xz, W_hz, b_z = triple()  # Update gate parameters
    W_xr, W_hr, b_r = triple()  # Reset gate parameters
    W_xh, W_hh, b_h = triple()  # Candidate hidden state parameters
    W_hq, b_q = normal((num_hiddens, num_outputs)), torch.zeros(num_outputs, device=device)

    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_gru_state(batch_size, num_hiddens, device):
    return torch.zeros((batch_size, num_hiddens), device=device),


def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    outputs, (H,) = [], state
    for X in inputs:
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)


if __name__ == '__main__':
    batch_size, num_steps = 32, 35
    train_iter, vocab = rnn.load_data_time_machine(batch_size, num_steps)

    vocab_size, num_hiddens, num_epochs, lr = len(vocab), 256, 500, 1
    net = rnn.RNNModelScratch(len(vocab), num_hiddens, tool.try_gpu(), get_params, init_gru_state, gru)
    rnn.train(net, train_iter, vocab, lr, num_epochs, tool.try_gpu())
