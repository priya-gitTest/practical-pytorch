import torch
import math
from model import *

class CMDHandler:
    def run(self, in_stream):
        cmds = in_stream.read()
        idx = 0
        while idx < len(cmds):
            if cmds[idx] == 'o':
                idx += 1
                print(self.add_to_history(cmds[idx]))
            elif cmds[idx] == 'q':
                idx += 1
                print('%f' % self.log2_prob(cmds[idx]))
            elif cmds[idx] == 'g':
                c = self.generate()
                print(c)
                self.add_to_history(c)
            elif cmds[idx] == 'x':
                break
            else:
                raise Exception('invalid cmd')
            idx += 1

    def add_to_history(self, c):
        return 'add to history %s' % c

    def log2_prob(self, c):
        return 1

    def generate(self):
        return 'g'


class RNNCMDHandler(CMDHandler):
    def __init__(self, model):
        self.model = model
        self.neg_log_prob_func = torch.nn.CrossEntropyLoss()

        self.hidden = self.model.init_hidden(1)
        self.out = None
        self.add_to_history(RNN.kSOTChar)

    def add_to_history(self, c):
        self.out, self.hidden = self.evaluate(c)
        return 'added %s' % c#self.out, self.hidden

    def evaluate(self, c):
        ord_idx=np.array([[ord(c)]])
        inp = char_tensor(ord_idx)
        out, hidden = self.model(inp, self.hidden)
        return out, hidden

    def log2_prob(self, c):
        target = char_tensor(np.array([ord(c)]))
        return -self.neg_log_prob_func(self.out.view(1, -1), target) / math.log(2)

    def generate(self):
        output_dist = self.out.view(-1).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        c = chr(top_i.cpu())
        self.add_to_history(c)
        return c


if __name__ == '__main__':
    import sys
    import numpy as np

    seed = int(sys.argv[1])
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    hidden_size = 512
    n_layers = 1
    n_characters = 2**16+1

    model_fn = 'mytraining_new.pt'
    model = RNN(n_characters, hidden_size, n_characters, n_layers)
    model.load_state_dict(torch.load(model_fn))
    model = cuda_if_available(model)

    cmd_handler = RNNCMDHandler(model)
    cmd_handler.run(sys.stdin)


