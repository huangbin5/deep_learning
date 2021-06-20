from _tools import mini_tool as tool
from _tools import rnn_frame as rnn
from _tools import att_frame as att

if __name__ == '__main__':
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps, lr, num_epochs = 64, 10, 0.005, 300
    train_iter, src_vocab, tgt_vocab = rnn.load_data_nmt(batch_size, num_steps)

    encoder = att.Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = att.Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    net = att.EncoderDecoder(encoder, decoder)
    att.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, tool.try_gpu())

    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, _ = att.predict_seq2seq(net, eng, src_vocab, tgt_vocab, num_steps, tool.try_gpu())
        print(f'{eng} => {translation}, bleu {att.bleu(translation, fra, k=2):.3f}')
