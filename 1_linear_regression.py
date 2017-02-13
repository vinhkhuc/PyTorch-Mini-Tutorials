import torch
from torch.autograd import Variable
from torch import optim


def build_model():
    model = torch.nn.Sequential()
    model.add_module("linear", torch.nn.Linear(1, 1, bias=False))
    return model


def train(model, loss, optimizer, x, y):
    x = Variable(x, requires_grad=False)
    y = Variable(y, requires_grad=False)

    # Reset gradient
    optimizer.zero_grad()

    # Forward
    fx = model.forward(x.view(len(x), 1))
    output = loss.forward(fx, y)

    # Backward
    output.backward()

    # Update parameters
    optimizer.step()

    return output.data[0]


def main():
    torch.manual_seed(42)
    X = torch.linspace(-1, 1, 101)
    Y = 2 * X + torch.randn(X.size()) * 0.33

    model = build_model()
    loss = torch.nn.MSELoss(size_average=True)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    batch_size = 10

    for i in range(100):
        cost = 0.
        num_batches = len(X) / batch_size
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            cost += train(model, loss, optimizer, X[start:end], Y[start:end])
        print("Epoch = %d, cost = %s" % (i + 1, cost / num_batches))

    w = model.parameters().next().data  # model has only one parameter
    print("w = %.2f" % w.numpy())  # will be approximately 2

if __name__ == "__main__":
    main()
