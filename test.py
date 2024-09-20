import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import time
import os

def backward_hook(module, output_grad):
    torch.cuda.empty_cache()
    return output_grad

def forward_hook(module, inputs, outputs):
    torch.cuda.empty_cache()

    
def pre_forward_hook(module, inputs):
    torch.cuda.empty_cache()
    return inputs
    
class Modified_Linear(nn.Module):
    def __init__(self, in_features, out_features, name):
        self.linear = nn.Linear(in_features, out_features)
        self.name = name

    def forward(self, x):
        file_name = os.path.join("computational_graph", self.name)
        if os.path.isfile(file_name):
            return torch.load(file_name, weights_only = True)
        else:
            return self.linear(x)

class ggnore(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Modified_Linear(10000, 10000, "fc1.pt")
        self.fc2 = Modified_Linear(10000, 10000, "fc2.pt")
        self.fc3 = Modified_Linear(10000, 10000, "fc3.pt")

        self.fc1.register_full_backward_pre_hook(backward_hook)
        self.fc2.register_full_backward_pre_hook(backward_hook)
        self.fc3.register_full_backward_pre_hook(backward_hook)

        self.fc1.register_forward_hook(forward_hook)
        self.fc2.register_forward_hook(forward_hook)
        self.fc3.register_forward_hook(forward_hook)

        self.fc1.register_forward_pre_hook(pre_forward_hook)
        self.fc2.register_forward_pre_hook(pre_forward_hook)
        self.fc3.register_forward_pre_hook(pre_forward_hook)

    def forward(self, x):
        def forward_func(x):
            x = self.fc1(x) + self.fc2(x)
            x = x + self.fc2(x)
            x = x + self.fc2(x)
            x = x + self.fc2(x)
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
            return x

        return checkpoint(forward_func, x, use_reentrant = False)
    
model = ggnore()
model.to("cuda")

input = torch.randn(10000, 10000, device = "cuda")
output = model(input)
output = output.sum()

