import time
import torch
from video import exist_video, delete_video, download_video
from model import exist_model_in_drive, Diffusion_Video_Model

torch.set_printoptions(
    precision = 4,
    sci_mode = False,
    threshold = 100
)

model = Diffusion_Video_Model()
if torch.cuda.is_available():
    model.cuda()

for i in range(1000000):
    if exist_video():
        delete_video()
    download_video()
    time.sleep(10)
    if exist_model_in_drive():
        model.load()
    else:
        create_model_in_drive(model)

    model.train()
    model.add_gradient_to_model_in_drive()


    
    
