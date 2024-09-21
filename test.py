import gc
import sys
import torch
a = torch.randn(4, 3, 384, 512)
print(sys.getsizeof(a.storage()))