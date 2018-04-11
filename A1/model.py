import torch
import torch.nn as nn
from torch.autograd import Variable

def cuda_if_available(v):
    return v.cuda() if torch.cuda.is_available() else v

class RNN(nn.Module):
    kSOTChar = '\u0002'
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, batch_first = True)
        self.decoder = nn.Linear(hidden_size, output_size)

    # output:(batch, seq_len, out_size),
    # hidden: (num_layers * num_directions, batch, hidden_size)
    def forward(self, input, hidden):
        input = self.encoder(input) # (B, seq_len, embedding_dim)
        output, hidden = self.gru(input, hidden) # output:(batch, seq_len, hidden_size * num_directions),
        output = self.decoder(output)
        return output, hidden

    def init_hidden(self, batch_size):
        v = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        return cuda_if_available(v)

# Turn string into list of longs
def char_tensor(ords_batch):
    tensor = torch.from_numpy(ords_batch).long()
    v = Variable(tensor)
    return cuda_if_available(v)

# size: (B, seq_len)
def random_training_set():
    while True:
        chunk = random_chunk()
        if chunk.shape[1] > 1:
            break
    inp = char_tensor(chunk[:,:-1])
    target = char_tensor(chunk[:,1:])
    mask_np = (chunk[:,1:] != kPadCharIndex).astype(np.float32)
    mask = Variable(torch.from_numpy(mask_np), requires_grad = False)

    return inp, target, cuda_if_available(mask)

