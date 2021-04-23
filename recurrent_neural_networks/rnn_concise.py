from torch import nn
from _tools import mini_tool as tool
from _tools import rnn_frame as rnn

if __name__ == '__main__':
    batch_size, num_steps, num_hiddens = 32, 35, 256
    train_iter, vocab = rnn.load_data_time_machine(batch_size, num_steps)

    num_epochs, lr = 500, 1
    rnn_layer = nn.RNN(len(vocab), num_hiddens)
    net = rnn.RNNModel(rnn_layer, vocab_size=len(vocab))
    net = net.to(tool.try_gpu())
    rnn.train(net, train_iter, vocab, lr, num_epochs, tool.try_gpu())
