import gc
import sys
import torch
from torch import nn
from torchviz import make_dot

a = torch.randn(2, 4)
ggnore, ggnora = a
ggnore[0] = 100
print(a)
# class multiplus(torch.autograd.Function):
#     def forward(context, input1, input2):
#         context.save_for_backward(input1, input2)
#         return input1 * input2
        
#     def backward(context, output_gradient):
#         input1, input2 = context.saved_tensors
#         print(input1 * input2)

#         return output_gradient * input2, output_gradient * input1

# a = torch.randn(3, requires_grad=True)
# b = torch.randn(3, requires_grad=True)

# c = multiplus.apply(a, b)
# print(c)
        

# class ggnore(torch.autograd.Function):
#     def forward(context, module, input_tensor):
#         context.mod = module
#         return module.net(input_tensor)
        
#     def backward(context, output_gradient):
#         with torch.enable_grad():
#             input_tensor = t
#             print(ggnora * 2)
#         return None, None

# model = torch.tensor([], requires_grad = True)
# model.net = nn.Linear(10, 10)
# input_tensor = torch.randn(2, 10)
# c = ggnore.apply(model, input_tensor)
# d = c.sum()
# d.backward()

# a = torch.tensor(9., requires_grad=True)
# b = torch.tensor(10., requires_grad=True)

# c = a * b
# c.backward(torch.tensor(10.))
# print(a.grad)