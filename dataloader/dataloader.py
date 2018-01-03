import numpy as np
import torch as t
from torch.autograd import Variable


class Dataloader():
    def __init__(self, data_path=''):
        """
        :param data_path: path to data
        """

        assert isinstance(data_path, str), \
            'Invalid data_path type. Required {}, but {} found'.format(str, type(data_path))

        self.data_path = data_path

        self.langs = ['en', 'ru']

        self.data_files = {
            lang: {
                'train': data_path + '{}_train_idx.txt'.format(lang),
                'valid': data_path + '{}_valid_idx.txt'.format(lang)
            }
            for lang in self.langs
        }
        self.model_files = {
            lang: {
                'model': data_path + '{}.model'.format(lang),
                'vocab': data_path + '{}.vocab'.format(lang)
            }
            for lang in self.langs
        }

        '''
        Special symbols are obtained for each vocabulary
        And their indexes are choosen in such way that they are unique
        '''
        vocab_size = {lang: len(open(self.model_files[lang]['vocab'], "r").read().split('\n')) - 1
                      for lang in self.langs}
        self.go_idx, self.stop_idx, self.pad_idx = ({lang: vocab_size[lang] + i for lang in self.langs}
                                                    for i in range(3))
        self.vocab_size = {
            lang: self.pad_idx[lang] + 1
            for lang in self.langs
        }

        self.data = {
            lang: {
                target: [[int(idx) for idx in line.split()]
                         for line in open(self.data_files[lang][target], "r").read().split('\n')[:-1]]
                for target in ['train', 'valid']
            }
            for lang in self.langs
        }

        self.max_len = {
            lang: max([len(line) + 2 for target in ['train', 'valid'] for line in self.data[lang][target]])
            for lang in self.langs
        }

        print('Data have loaded')

    def next_batch(self, batch_size, target: str):
        """
        :param batch_size: Number of selected data elements
        :param target: 'train' or 'valid'
        :return: Target ndarrays
        """

        indexes = np.array(np.random.randint(len(self.data['en'][target]), size=batch_size))
        lines = {lang: [self.data[lang][target][index] for index in indexes] for lang in self.langs}

        return self.construct_batches(lines)

    def construct_batches(self, lines):
        """
        :param lines: An dict of indexes arrays
        :return: Batches
        """

        condition = lines['en']
        input = [[self.go_idx['ru']] + line for line in lines['ru']]
        target = [line + [self.stop_idx['ru']] for line in lines['ru']]

        condition = self.padd_sequences(condition, lang='en')
        input = self.padd_sequences(input, lang='ru')
        target = self.padd_sequences(target, lang='ru')

        return condition, input, target

    def padd_sequences(self, lines, lang):

        lengths = [len(line) for line in lines]
        max_length = max(lengths)

        return np.array([line + [self.pad_idx[lang]] * (max_length - lengths[i])
                         for i, line in enumerate(lines)])

    def torch(self, batch_size, target, cuda, volatile=False):

        condition, input, target = self.next_batch(batch_size, target)
        condition, input, target = [Variable(t.from_numpy(var), volatile=volatile)
                                    for var in [condition, input, target]]
        if cuda:
            condition, input, target = [var.cuda() for var in [condition, input, target]]

        return condition, input, target

    def go_input(self, batch_size, cuda, lang, volatile=True):

        tensor = Variable(t.LongTensor([[self.go_idx[lang]]] * batch_size), volatile=volatile)
        if cuda:
            tensor = tensor.cuda()

        return tensor

    def to_tensor(self, lines, cuda, lang, volatile=True):

        tensor = Variable(t.LongTensor([[self.go_idx[lang]] + line for line in lines]), volatile=volatile)
        if cuda:
            tensor = tensor.cuda()

        return tensor
