import numpy as np


class ScheduledOptim(object):
    def __init__(self, optimizer, d_model, n_layers, n_warmup_steps):

        self.optimizer = optimizer

        self.d_model = d_model
        self.n_layers = n_layers
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def update_learning_rate(self):
        self.n_current_steps += 1

        model_lr = np.power(self.d_model, -0.5) * np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

        fine_lr = np.power(self.d_model / self.n_layers, -0.5) * np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps / 10])

        for param_group in self.optimizer.param_groups:
            if not param_group['fine']:
                param_group['lr'] = model_lr
            else:
                param_group['lr'] = fine_lr
