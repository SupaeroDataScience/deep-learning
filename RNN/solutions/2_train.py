# training step through all data
def closure():
    optimizer.zero_grad()
    hidden=rnn.init_hidden()
    out, hidden = rnn(input, hidden, future=10)
    loss = criterion(out[:, 10:-10], target[:, 10:])
    print('loss:', loss.item())
    loss.backward()
    return loss