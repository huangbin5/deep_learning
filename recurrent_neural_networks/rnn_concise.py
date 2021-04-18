import torch
from torch import nn
from torch.nn import functional as F
from _tools import cnn_frame as cnn
from _tools import rnn_frame as rnn


class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super().__init__()
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # 如果RNN是双向的，`num_directions` = 2，否则 = 1。
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size).to(torch.float32)
        Y, state = self.rnn(X, state)
        # Y.shape = (time_step, batch_size, num_hidden)
        outputs = self.linear(Y.reshape((-1, Y.shape[-1])))
        return outputs, state

    def begin_state(self, device, batch_size=1):
        # GRU返回tensor
        state = torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device)
        if isinstance(self.rnn, nn.LSTM):
            # LSTM返回Tuple
            state = (state, state)
        return state


if __name__ == '__main__':
    batch_size, num_steps, num_hiddens = 32, 35, 256
    train_iter, vocab = rnn.load_data_time_machine(batch_size, num_steps)
    state = torch.zeros((1, batch_size, num_hiddens))
    rnn_layer = nn.RNN(len(vocab), num_hiddens)

    net = RNNModel(rnn_layer, vocab_size=len(vocab))
    net = net.to(cnn.try_gpu())
    num_epochs, lr = 500, 1
    rnn.train(net, train_iter, vocab, lr, num_epochs, cnn.try_gpu())
