import torch
from _tools import cnn_frame as cnn
from _tools import rnn_frame as rnn


def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size
    normal = lambda shape: torch.randn(size=shape, device=device) * 0.01
    triple = lambda: (normal((num_inputs, num_hiddens)), normal((num_hiddens, num_hiddens)),
                      torch.zeros(num_hiddens, device=device))

    W_xi, W_hi, b_i = triple()  # Input gate parameters
    W_xf, W_hf, b_f = triple()  # Forget gate parameters
    W_xo, W_ho, b_o = triple()  # Output gate parameters
    W_xc, W_hc, b_c = triple()  # Candidate memory cell parameters
    W_hq, b_q = normal((num_hiddens, num_outputs)), torch.zeros(num_outputs, device=device)

    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))


def lstm(inputs, state, params):
    W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q = params
    outputs, (H, C) = [], state
    for X in inputs:
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * torch.tanh(C)
        Y = (H @ W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)


if __name__ == '__main__':
    batch_size, num_steps = 32, 35
    train_iter, vocab = rnn.load_data_time_machine(batch_size, num_steps)

    vocab_size, num_hiddens, num_epochs, lr = len(vocab), 256, 500, 1
    net = rnn.RNNModelScratch(len(vocab), num_hiddens, cnn.try_gpu(), get_lstm_params, init_lstm_state, lstm)
    rnn.train(net, train_iter, vocab, lr, num_epochs, cnn.try_gpu())
