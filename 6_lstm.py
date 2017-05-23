from __future__ import division
import numpy as np

import torch
from torch.autograd import Variable
from torch import optim, nn

from data_util import load_mnist


class LSTMNet(torch.nn.Module):
    def __init__(self, batch_size, input_dim, hidden_dim, output_dim):
        super(LSTMNet, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        h0 = Variable(torch.zeros([1, self.batch_size, self.hidden_dim]), requires_grad=False)
        c0 = Variable(torch.zeros([1, self.batch_size, self.hidden_dim]), requires_grad=False)
        fx, hn = self.lstm.forward(x, (h0, c0))
        return self.linear.forward(fx[-1])


def train(model, loss, optimizer, x_val, y_val):
    x = Variable(x_val, requires_grad=False)
    y = Variable(y_val, requires_grad=False)

    # Reset gradient
    optimizer.zero_grad()

    # Forward
    fx = model.forward(x)
    output = loss.forward(fx, y)

    # Backward
    output.backward()

    # Update parameters
    optimizer.step()

    return output.data[0]


def predict(model, x_val):
    n_samples = x_val.size()[1]
    batch_size = model.batch_size
    res = np.empty(n_samples)
    for i in range(n_samples // batch_size):
        l, r = i * batch_size, (i + 1) * batch_size
        x = Variable(x_val[:, l:r, :], requires_grad=False)
        output = model.forward(x)
        res[l:r] = output.data.numpy().argmax(axis=1)
    return res


def main():
    torch.manual_seed(42)
    trX, teX, trY, teY = load_mnist(onehot=False)
    trX = trX.reshape(-1, 28, 28)
    teX = teX.reshape(-1, 28, 28)

    # Convert to the shape (seq_length, num_samples, input_dim)
    trX = np.swapaxes(trX, 0, 1)  # (28, -1, 28)
    teX = np.swapaxes(teX, 0, 1)  # (28, -1, 28)

    trX = torch.from_numpy(trX).float()
    teX = torch.from_numpy(teX).float()
    trY = torch.from_numpy(trY).long()

    batch_size = 100
    n_classes = 10
    input_dim = 28
    hidden_dim = 128
    model = LSTMNet(batch_size, input_dim, hidden_dim, n_classes)
    loss = torch.nn.CrossEntropyLoss(size_average=True)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for i in range(20):
        cost = 0.
        num_batches = trX.size()[1] // batch_size
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            cost += train(model, loss, optimizer, trX[:, start:end, :], trY[start:end])
        predY = predict(model, teX)
        print("Epoch %d, cost = %f, acc = %.2f%%" %
              (i + 1, cost / num_batches, 100. * np.mean(predY == teY)))


if __name__ == "__main__":
    main()
