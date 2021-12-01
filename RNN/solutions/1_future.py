class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn1 = nn.RNNCell(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, future=0):
        # predict over the different signals
        outputs = []
        for input_t in input.split(1, dim=1):
            hidden = self.rnn1(input_t, hidden)
            output = self.linear(hidden)
            outputs += [output]
        for i in range(future):# if we should predict the future
            hidden = self.rnn1(output, hidden)
            output = self.linear(hidden)
            outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs, hidden
    
    def init_hidden(self):
        return torch.zeros(input.size(0), self.hidden_size, dtype=torch.double)