import torch.nn as nn
from torch.nn.init import xavier_normal


class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(Embedding, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.embeddings.weight = xavier_normal(self.embeddings.weight)

    def forward(self, input):
        return self.embeddings(input)
