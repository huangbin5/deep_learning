import torch
from _tools import mini_tool as tool
from _tools import rnn_frame as rnn
from _tools import att_frame as att

if __name__ == '__main__':
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps, lr, num_epochs = 64, 10, 0.005, 250
    train_iter, src_vocab, tgt_vocab = rnn.load_data_nmt(batch_size, num_steps)

    encoder = att.Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = att.Seq2SeqAttentionDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    net = att.EncoderDecoder(encoder, decoder)
    att.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, tool.try_gpu())

    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, dec_attention_weight_seq = att.predict_seq2seq(net, eng, src_vocab, tgt_vocab, num_steps,
                                                                    tool.try_gpu(), True)
        print(f'{eng} => {translation}, ', f'bleu {att.bleu(translation, fra, k=2):.3f}')

    attention_weights = torch.cat([step[0][0][0] for step in dec_attention_weight_seq],
                                  0).reshape((1, 1, -1, num_steps))
    att.show_heatmaps(attention_weights[:, :, :, :len(engs[-1].split()) + 1].cpu(),
                      xlabel='Key posistions', ylabel='Query posistions')
