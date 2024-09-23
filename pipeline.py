import torch
from video import exist_video, delete_video, download_video
from model import exist_model, Diffusion_Video_Model, show_image, make_video
import os

torch.set_printoptions(
    precision = 4,
    sci_mode = False,
    threshold = 100
)

if not os.path.exists("computational_graph"):
    os.mkdir("computational_graph")

model = Diffusion_Video_Model()
if exist_model():
    model.load()

if torch.cuda.is_available():
    model.cuda()
    print("Model has been moved to CUDA!")

print("Started to train autoencoder...")
for time_step in range(1000000):
    if exist_video():
        delete_video()
    download_video(time_step, "Autoencoder")

    loss = model.train_auto_encoder(time_step, 2)
    print("Time step " + str(time_step) + ": Autoencoder Loss = " + str(loss))
    model.save()

print("Started to train stable diffusion...")
for time_step in range(1000000):
    if  exist_video():
        delete_video()
    download_video(time_step, "Stable Diffusion")

    loss = model.train_stable_diffusion(time_step, 100)
    print("Time step " + str(time_step) + ": Stable Diffusion Loss = " + str(loss))
    model.save()

print("Started inference...")

batch_video, _ = model.infer([
    "I eat shit",
    "I eat cock"
], (64, 96), 10)

make_video(batch_video)
show_image(batch_video[0][0])