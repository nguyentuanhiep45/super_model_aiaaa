import time
import torch
from video import exist_video, delete_video, download_video
from model import exist_model, Diffusion_Video_Model

torch.set_printoptions(
    precision = 4,
    sci_mode = False,
    threshold = 100
)

model = Diffusion_Video_Model()
if exist_model():
    model.load()

if torch.cuda.is_available():
    model.cuda()

for time_step in range(1000000):
    if exist_video():
        delete_video()
    download_video(time_step)

    model.train(time_step)
    model.save()