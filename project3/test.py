import torch

print(torch.rand((3, 2)) * torch.tensor([3, 2.]) + torch.tensor([1., -1.]))
t = torch.rand((1000, 3, 2)) * torch.tensor([3, 2.]) + torch.tensor([1., -1.])
print(t[0])


