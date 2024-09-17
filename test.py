import torch
from torch import nn
from model import Diffusion_Video_Model, show_image

torch.set_printoptions(
    precision = 4,
    sci_mode = False,
    threshold = 100
)

model = Diffusion_Video_Model()
if torch.cuda.is_available():
    model.cuda()

model.one_step_train(
    torch.randn(2, 4, 3, 128, 128, device = model.device),
    [
        "I eat shit",
        "I love you"
    ]
)
