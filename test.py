from function_low_gpu import Modified_Multiply
import torch

a = 15.6
b = torch.randn(3, 4, requires_grad = True)
c = a * b

d = c.sum()
print(c)