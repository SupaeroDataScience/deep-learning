class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn1 = nn.RNNCell(1, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, inp, hidden, future=0):
        outputs = []
        for signal in inp.split(1, dim=1):
            hidden = self.rnn1(signal, hidden)
            output = self.linear(hidden)
            outputs += [output]
        for i in range(1, future):# if we should predict the future
            hidden = self.rnn1(output, hidden)
            output = self.linear(hidden)
            outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs, hidden

    def init_hidden(self):
        return torch.zeros(self.input_size, self.hidden_size, dtype=torch.double)
