import torch

alpha=torch.tensor([1.0,1.0,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])
print(torch.nn.functional.softmax(alpha,0))


