import torch
from model import Diffusion_Video_Model

torch.set_printoptions(
    precision = 4,
    sci_mode = False,
    threshold = 100
)

model = Diffusion_Video_Model()

if exist_model_in_drive():
    model.load()

model.test("man eat shit", (1920, 1080))