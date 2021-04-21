from torch import nn
from _tools import cnn_frame as cnn
from _tools import rnn_frame as rnn

if __name__ == '__main__':
    batch_size, num_steps = 32, 35
    train_iter, vocab = rnn.load_data_time_machine(batch_size, num_steps)

    num_inputs, num_hiddens, num_layers, num_epochs, lr = len(vocab), 256, 2, 500, 2
    lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
    net = rnn.RNNModel(lstm_layer, len(vocab))
    net = net.to(cnn.try_gpu())
    rnn.train(net, train_iter, vocab, lr, num_epochs, cnn.try_gpu())
