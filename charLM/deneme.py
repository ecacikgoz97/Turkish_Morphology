import torch

a = torch.tensor([[2., 1.], [5., 10.]])
b = torch.tensor([[8., 2.], [3., 14.]])
print(f"a: {a.shape}\n{a}")
print(f"b: {b.shape}\n\n{b}")

ab1 = a @ b.T
print(f"ab1: {ab1.shape}\n{ab1}")
ab2 = torch.mm(a, b.T)
print(f"ab2: {ab2.shape}\n{ab2}\n\n")


ab1 = torch.mm(a, b.T)
print(f"ab1: {ab1.shape}\n{ab1}")
ab2 = torch.mm(a, b).transpose(-2, -1)
print(f"ab2: {ab2.shape}\n{ab2}")
