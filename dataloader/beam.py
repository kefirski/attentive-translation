class Beam:
    def __init__(self, prob=1., data=''):
        self.prob = prob
        self.data = data

    def update(self, prob, token):
        self.prob = prob
        self.data += token
        return self

    def __str__(self):
        return 'p = {}, data = {}'.format(self.prob, self.data)
