class skipgram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        #vocab size : vocabulary size (corresponding to input and output dimensions)
        #embedding_dim : dimension of the embedding (hidden) layer
        super(skipgram, self).__init__()
        self.lin1 = nn.Linear(vocab_size,embedding_dim)
        self.lin2 = nn.Linear(embedding_dim,vocab_size)
        self.soft= nn.Softmax(dim=0)
    def forward(self, input):
        hidden = self.lin1(input)
        output=self.lin2(hidden)
        output=self.soft(output)
        return output
    def get_wv(self,input):
        #get the word vector for a given input
        return self.lin1(input).detach().numpy()