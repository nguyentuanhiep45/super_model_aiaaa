import torch
from model import Diffusion_Video_Model

torch.set_printoptions(
    precision = 16,
    sci_mode = False,
    threshold = 100
)

model = Diffusion_Video_Model()

model.infer([
    "I eat shit",
    "I love you"
])