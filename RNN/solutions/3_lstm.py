class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTMCell(1, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, inp, h_t, c_t, future=0):
        outputs = []
        for signal in inp.split(1, dim=1):
            h_t, c_t = self.lstm1(signal, (h_t, c_t))
            output = self.linear(h_t)
            outputs += [output]
        for i in range(1, future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            output = self.linear(h_t)
            outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs, h_t, c_t

    def init_hidden(self):
        return (torch.zeros(self.input_size, self.hidden_size, dtype=torch.double),
                torch.zeros(self.input_size, self.hidden_size, dtype=torch.double))

