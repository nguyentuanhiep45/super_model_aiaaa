import torch
from model import Diffusion_Video_Model, exist_model, show_image

torch.set_printoptions(
    precision = 4,
    sci_mode = False,
    threshold = 100
)

model = Diffusion_Video_Model()

if exist_model():
    model.load()

batch_video, _ = model.infer([
    "I eat shit",
    "I eat cock"
], (64, 96), 10)

show_image(batch_video[0][0])