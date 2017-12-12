from math import sqrt

import torch as t
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, size, m_size=None, p=0.1):
        """
        :param size: float number that is necessary for estimation scaling factor
        :param m_size: int number of size of the window that performing local-m attention.
                   None corresponds to global attention mechanism
        :param p: drop prob
        """
        super(ScaledDotProductAttention, self).__init__()

        self.m_size = m_size

        self.scaling = 1 / (sqrt(size))
        self.dropout = nn.Dropout(p)

    def forward(self, q, k, v, mask=None):
        """
        :param q: An float tensor with shape of [batch_size, query_len, size]
        :param k: An float tensor with shape of [batch_size, seq_len, size]
        :param v: An float tensor with shape of [batch_size, seq_len, value_size]
        :param mask: An byte tensor with shape of [batch_size, query_len, seq_len]
        :return: An float tensor with shape of [batch_size, query_len, value_size]
                     and attention map with shape of [batch_size, query_len, seq_len]
        """

        batch_size, query_len, _ = q.size()

        attention = t.bmm(q, k.transpose(1, 2)) * self.scaling

        '''
        In order to prevent contribution of padding symbols in attention lockup, 
        it is necessary to use attention mask
        
        There is no problem in the case when only one mask is applied,
        however, if m-local mask is applied with padding mask,
        then NaN will arice, since softmax is undefined for array filled with -inf values
        
        The workaround for this issue is to drop masks for such ill rows
        '''
        if mask is not None and self.m_size is None:
            attention.data.masked_fill_(mask, -float('inf'))

        elif mask is None and self.m_size is not None:
            attention.data.masked_fill_(self.window_mask(batch_size, query_len, q.is_cuda), -float('inf'))

        elif mask is not None and self.m_size is not None:
            window = self.window_mask(batch_size, query_len, q.is_cuda)
            mask = t.ge(window + mask, 1)

            mask_intersection = t.eq(mask.min(2)[0], 1).unsqueeze(2).repeat(1, 1, query_len)
            mask.masked_fill_(mask_intersection, 0)

            attention.data.masked_fill_(mask, -float('inf'))

        attention = F.softmax(attention, dim=2)

        return t.bmm(self.dropout(attention), v), attention

    def window_mask(self, batch_size, seq_len, use_cuda):

        size = 2 * self.m_size + 1 if self.m_size is not None else seq_len

        if seq_len <= size:
            mask = t.zeros(batch_size, seq_len, seq_len).byte()
            if use_cuda:
                mask = mask.cuda()

            return mask
        else:

            result = [(0, size)] * self.m_size

            for i in range(self.m_size, seq_len - self.m_size):
                result += [(i - self.m_size, i + self.m_size + 1)]

            result += [(seq_len - 2 * self.m_size - 1, seq_len)] * self.m_size

        mask = t.ones(batch_size, seq_len, seq_len).byte()
        for i, (a, b) in enumerate(result):
            mask[:, i, a:b] = t.zeros(batch_size, 1, size)

        if use_cuda:
            mask = mask.cuda()

        return mask
