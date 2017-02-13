import torch

a = torch.IntTensor([2, 3, 4])
b = torch.IntTensor([3, 4, 5])
m = a * b  # element-wise product
print(m.numpy())  # convert to the numpy array [ 6 12 20]
