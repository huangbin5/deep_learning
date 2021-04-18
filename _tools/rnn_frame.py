import re
import collections
import torch
import random
import math
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
from _tools.constant import *
from _tools import mlp_frame as mlp
from _tools import mini_tool as tool


def read_time_machine():
    DATA_HUB['time_machine'] = (DATA_URL + 'timemachine.txt', '090b5e7e70c295757f55df93cb0a180b9691891a')
    with open(tool.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


def tokenize(lines, token='word'):
    """将文本行拆分为单词或字符"""
    assert token in ['word', 'char'], 'unknown token type'
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]


def count_corpus(tokens):
    """统计语料库"""
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 统计语料库并按词频排序
        counter = count_corpus(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # 未知标记的索引为0
        self.unk, uniq_tokens = 0, ['<unk>'] + reserved_tokens
        uniq_tokens += [token for token, freq in self.token_freqs if freq >= min_freq and token not in uniq_tokens]
        self.idx_to_token, self.token_to_idx = [], dict()
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]


def load_corpus_time_machine(max_tokens=-1):
    """将文本中的每个字符转化为索引"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


def seq_data_iter_random(corpus, batch_size, num_steps):
    """使用随机抽样生成一小批子序列。"""
    # 随机选择一个偏移开始
    corpus = corpus[random.randint(0, num_steps - 1):]
    num_subseqs = (len(corpus) - 1) // num_steps
    # 序列的所有起始位置
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 打乱顺序，即随机抽样
    random.shuffle(initial_indices)

    data = lambda pos: corpus[pos:pos + num_steps]
    num_batches = num_subseqs // batch_size
    for i in range(0, num_batches * batch_size, batch_size):
        # 该批次的起始位置
        initial_indices_per_batch = initial_indices[i:i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)


def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """使用顺序分区生成一小批子序列。"""
    # 随机选择一个偏移开始
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset:offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1:offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i:i + num_steps]
        Y = Ys[:, i:i + num_steps]
        yield X, Y


class SeqDataLoader:
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


class RNNModelScratch:
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        # 将`时间步数`维度移到前面，方便一步步更新隐藏状态
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


def predict(prefix, num_preds, net, vocab, device):
    """预测`prefix`之后的`num_preds`个字符"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    # 输入的time_step和batch_size均为1
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:  # 预热期
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 预测`num_preds`步
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


def grad_clipping(net, theta):
    # 梯度剪裁，缓解梯度爆炸
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def train_epoch(net, train_iter, cross_entropy, optimizer, device, use_random_iter):
    state, timer = None, tool.Timer()
    # (sum of loss, num of tokens)
    metric = tool.Accumulator(2)
    for X, Y in train_iter:
        if state is None or use_random_iter:
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # GRU时state是一个tensor
                state.detach_()
            else:
                # 自定义实现或LSTM中state是一个Tuple
                for s in state:
                    s.detach_()
        # 转换为 (time_step * batch_size)的一维向量
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        loss = cross_entropy(y_hat, y).mean()
        if isinstance(optimizer, torch.optim.Optimizer):
            optimizer.zero_grad()
            loss.backward()
            grad_clipping(net, 1)
            optimizer.step()
        else:
            loss.backward()
            grad_clipping(net, 1)
            optimizer(batch_size=1)
        metric.add(loss * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


def train(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    cross_entropy = nn.CrossEntropyLoss()
    if isinstance(net, nn.Module):
        optimizer = torch.optim.SGD(net.parameters(), lr)
    else:
        optimizer = lambda batch_size: mlp.sgd(net.params, lr, batch_size)
    predict_func = lambda prefix: predict(prefix, 50, net, vocab, device)
    animator = tool.Animator(xlabel='epoch', ylabel='perplexity', legend=['train'], xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        print('epoch {}/{} begin...'.format(epoch + 1, num_epochs))
        ppl, speed = train_epoch(net, train_iter, cross_entropy, optimizer, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict_func('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    print(predict_func('time traveller'))
    print(predict_func('traveller'))
    plt.show()
