import numpy as np

import torch
from torch.autograd import Variable
from torch import optim

from data_util import load_mnist


def build_model(input_dim, output_dim):
    model = torch.nn.Sequential()
    model.add_module("linear_1", torch.nn.Linear(input_dim, 512, bias=False))
    model.add_module("relu_1", torch.nn.ReLU())
    model.add_module("dropout_1", torch.nn.Dropout(0.2))
    model.add_module("linear_2", torch.nn.Linear(512, 512, bias=False))
    model.add_module("relu_2", torch.nn.ReLU())
    model.add_module("dropout_2", torch.nn.Dropout(0.2))
    model.add_module("linear_3", torch.nn.Linear(512, output_dim, bias=False))
    return model


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
    x = Variable(x_val, requires_grad=False)
    output = model.forward(x)
    return output.data.numpy().argmax(axis=1)


def main():
    torch.manual_seed(42)
    trX, teX, trY, teY = load_mnist(onehot=False)
    trX = torch.from_numpy(trX).float()
    teX = torch.from_numpy(teX).float()
    trY = torch.from_numpy(trY).long()

    n_examples, n_features = trX.size()
    n_classes = 10
    model = build_model(n_features, n_classes)
    loss = torch.nn.CrossEntropyLoss(size_average=True)
    optimizer = optim.Adam(model.parameters())
    batch_size = 100

    for i in range(100):
        cost = 0.
        num_batches = n_examples / batch_size
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            cost += train(model, loss, optimizer, trX[start:end], trY[start:end])
        predY = predict(model, teX)
        print("Epoch %d, cost = %f, acc = %.2f%%"
              % (i + 1, cost / num_batches, 100. * np.mean(predY == teY)))


if __name__ == "__main__":
    main()
