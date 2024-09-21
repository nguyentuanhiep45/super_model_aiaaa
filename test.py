import gc
import sys
import torch
from torchviz import make_dot
a = torch.randn(10, 10, requires_grad=True)
b = a + a
c = b + a
d = b * c
e = d + 2 * c
f = e.sum()

gr = make_dot(f, {
    "a": a,
    "b": b
})

gr.view("computational_graph")