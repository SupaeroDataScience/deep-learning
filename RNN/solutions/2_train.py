# training step through all data
def closure():
    optimizer.zero_grad()
    out, hidden = rnn(input, future=5)
    loss = criterion(out[:, :-10], target[:, 5:])
    print('loss:', loss.item())
    loss.backward()
    return loss
