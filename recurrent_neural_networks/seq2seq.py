from _tools import cnn_frame as cnn
from _tools import rnn_frame as rnn

if __name__ == '__main__':
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps, lr, num_epochs = 64, 10, 0.005, 300
    train_iter, src_vocab, tgt_vocab = rnn.load_data_nmt(batch_size, num_steps)

    encoder = rnn.Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = rnn.Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    net = rnn.EncoderDecoder(encoder, decoder)
    rnn.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, cnn.try_gpu())

    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, _ = rnn.predict_seq2seq(net, eng, src_vocab, tgt_vocab, num_steps, cnn.try_gpu())
        print(f'{eng} => {translation}, bleu {rnn.bleu(translation, fra, k=2):.3f}')