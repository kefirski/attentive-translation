import numpy as np
import torch as t
import torch.nn as nn
from gensim.models.keyedvectors import KeyedVectors
from torch.autograd import Variable


class PositionalEmbeddings(nn.Module):
    def __init__(self, path, vocab_size, max_len, h_size, padding_idx=0):
        super(PositionalEmbeddings, self).__init__()

        self.max_len = max_len
        self.embedding_size = h_size

        self.padding_idx = padding_idx

        self.token_embeddings = nn.Embedding(vocab_size, h_size, padding_idx=padding_idx)
        self.positional_embeddings = nn.Embedding(int(max_len), h_size, padding_idx=0)

        '''
        w2v model contains vectors for each index in vocabulary.
        Here we lockup them and add vectors for go, end and pad tokens
        '''
        keyed_vectors = KeyedVectors.load_word2vec_format(path, binary=True)
        embeddings = np.array([keyed_vectors.wv[str(idx)] if str(idx) in keyed_vectors.vocab else np.zeros([1, h_size])
                               for idx in range(vocab_size - 3)])
        embeddings = np.concatenate([embeddings, np.ones([3, h_size])], 0)
        self.token_embeddings.weight = nn.Parameter(t.from_numpy(embeddings), requires_grad=False)
        self.position_encoding_init()

    def forward(self, input):
        batch_size, seq_len = input.size()

        positional = Variable(t.LongTensor([i for i in range(1, seq_len + 1)])).repeat(batch_size).view(batch_size, -1)
        if input.is_cuda:
            positional = positional.cuda()

        return self.token_embeddings(input) + self.positional_embeddings(positional)

    def position_encoding_init(self):
        encoding = np.array([
            [pos / np.power(10000, 2 * i / self.embedding_size) for i in range(self.embedding_size)]
            if pos != 0 else np.zeros(self.embedding_size) for pos in range(self.max_len)])

        encoding[1:, 0::2] = np.sin(encoding[1:, 0::2])
        encoding[1:, 1::2] = np.cos(encoding[1:, 1::2])

        self.positional_embeddings.weight = nn.Parameter(t.from_numpy(encoding).float(), requires_grad=False)
