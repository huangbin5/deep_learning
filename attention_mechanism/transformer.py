import torch
import pandas as pd
from _tools import mini_tool as tool
from _tools import rnn_frame as rnn
from _tools import att_frame as att

num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
lr, num_epochs = 0.005, 200
ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
key_size, query_size, value_size = 32, 32, 32
norm_shape = [32]

train_iter, src_vocab, tgt_vocab = rnn.load_data_nmt(batch_size, num_steps)

encoder = att.TransformerEncoder(len(src_vocab), key_size, query_size, value_size, num_hiddens, norm_shape,
                                 ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout)
decoder = att.TransformerDecoder(len(tgt_vocab), key_size, query_size, value_size, num_hiddens, norm_shape,
                                 ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout)
net = att.EncoderDecoder(encoder, decoder)
att.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, tool.try_gpu())

engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = att.predict_seq2seq(net, eng, src_vocab, tgt_vocab, num_steps,
                                                                tool.try_gpu(), True)
    print(f'{eng} => {translation}, bleu {att.bleu(translation, fra, k=2):.3f}')

enc_attention_weights = torch.cat(net.encoder.attention_weights, 0).reshape((num_layers, num_heads, -1, num_steps))
att.show_heatmaps(enc_attention_weights.cpu(), xlabel='Key positions', ylabel='Query positions',
                  titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))

dec_attention_weights_2d = [head[0].tolist() for step in dec_attention_weight_seq
                            for attn in step for blk in attn for head in blk]
dec_attention_weights_filled = torch.tensor(pd.DataFrame(dec_attention_weights_2d).fillna(0.0).values)
dec_attention_weights = dec_attention_weights_filled.reshape((-1, 2, num_layers, num_heads, num_steps))
dec_self_attention_weights, dec_inter_attention_weights = dec_attention_weights.permute(1, 2, 3, 0, 4)
att.show_heatmaps(dec_self_attention_weights[:, :, :, :len(translation.split()) + 1], xlabel='Key positions',
                  ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))
att.show_heatmaps(dec_inter_attention_weights, xlabel='Key positions', ylabel='Query positions',
                  titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))
