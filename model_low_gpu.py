import torch
from torch import nn
from torch import optim
from torch.nn import functional as func
from torch.utils.checkpoint import checkpoint
import transformers
import matplotlib.pyplot as plt
import random
import os
import cv2 as cv
from video import configuration_at_time_step
import re
import gc
import time
import sys
from function_low_gpu import one_input_forward, three_input_forward

def exist_model():
    return os.path.isfile("model.ckpt")

# shape (số dòng, số cột) trả về => chu kỳ để dòng lặp lại = cycle dòng
def positional_encoder(shape, cycle):
    m = cycle ** (2 / shape[1])
    temp = torch.arange(0, shape[0], dtype = torch.float32).reshape(shape[0], 1) @ \
           (m ** -torch.arange(0, shape[1] // 2)).unsqueeze(0)
    
    return torch.cat((torch.sin(temp), torch.cos(temp)), 1)

def time_encoder(size, time_step):
    m = 2000 ** (2 / size)
    temp = m ** -torch.arange(0, size // 2) * time_step
    return torch.cat((torch.sin(temp), torch.cos(temp)), 0)

def show_image(integer_tensor_image):
    plt.imshow(integer_tensor_image.permute(1, 2, 0))
    plt.axis('off')
    plt.show()

class Diffusion_First_Unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.diffusion_first_unit_layer = nn.ModuleList([
            nn.GroupNorm(32, in_channels),
            nn.Conv2d(in_channels, out_channels, 3, padding = 1),
            nn.Linear(4 * 320, out_channels),
            nn.GroupNorm(32, out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding = 1),
            nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
        ])

    def forward(self, latent, time_encoding):
        residue = latent
        latent = self.diffusion_first_unit_layer[0](latent)
        latent = func.silu(latent)
        latent = self.diffusion_first_unit_layer[1](latent)

        time_encoding = func.silu(time_encoding)
        latent = latent + self.diffusion_first_unit_layer[2](time_encoding).reshape(1, self.out_channels, 1, 1)
        latent = self.diffusion_first_unit_layer[3](latent)
        latent = func.silu(latent)
        latent = self.diffusion_first_unit_layer[4](latent) + self.diffusion_first_unit_layer[5](residue)

        return (latent, time_encoding)
        
class Diffusion_Second_Unit(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.out_channels = out_channels

        self.diffusion_second_unit_layer = nn.ModuleList([
            nn.GroupNorm(32, out_channels),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.LayerNorm(out_channels),
            nn.MultiheadAttention(out_channels, 8, batch_first = True),
            nn.LayerNorm(out_channels),
            nn.MultiheadAttention(out_channels, 8, kdim = 768, vdim = 768, batch_first = True),
            nn.LayerNorm(out_channels),
            nn.MultiheadAttention(out_channels, 8, kdim = 64, vdim = 64, batch_first = True),
            nn.LayerNorm(out_channels),
            nn.Linear(out_channels, out_channels * 8),
            nn.Linear(4 * out_channels, out_channels),
            nn.Conv2d(out_channels, out_channels, 1)
        ])

    def forward(self, latent, context, memory_latent):
        residue_long = latent
        latent = self.diffusion_second_unit_layer[0](latent)
        latent = self.diffusion_second_unit_layer[1](latent)
        h, w = latent.shape[-2:]
        latent = latent.reshape(-1, h * w, self.out_channels)
        residue_short = latent
        latent = self.diffusion_second_unit_layer[2](latent)
        latent = self.diffusion_second_unit_layer[3](latent, latent, latent)[0] + residue_short

        residue_short = latent
        latent = self.diffusion_second_unit_layer[4](latent)
        latent = self.diffusion_second_unit_layer[5](latent, context, context)[0] + residue_short

        residue_short = latent
        latent = self.diffusion_second_unit_layer[6](latent)
        latent = self.diffusion_second_unit_layer[7](latent, memory_latent, memory_latent)[0] + residue_short

        residue_short = latent
        latent = self.diffusion_second_unit_layer[8](latent)
        latent, gate = self.diffusion_second_unit_layer[9](latent).chunk(2, -1)
        latent = latent * func.gelu(gate)
        latent = self.diffusion_second_unit_layer[10](latent) + residue_short
        latent = latent.reshape(-1, self.out_channels, h, w)
        latent = self.diffusion_second_unit_layer[11](latent) + residue_long

        return latent

class Diffusion_Unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.diffusion_unit_layer = nn.ModuleList([
            Diffusion_First_Unit(in_channels, out_channels),
            Diffusion_Second_Unit(out_channels)
        ])

    def forward(self, latent, time_encoding, context, memory_latent):
        latent, time_encoding = self.diffusion_unit_layer[0](latent, time_encoding)
        return (self.diffusion_unit_layer[1](latent, context, memory_latent), time_encoding)

class Decoder_Unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.decoder_unit_layer = nn.ModuleList([
            nn.GroupNorm(32, in_channels),
            nn.Conv2d(in_channels, out_channels, 3, padding = 1),
            nn.GroupNorm(32, out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding = 1),
            nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
        ])

    def forward(self, latent):
        residue = latent
        latent = one_input_forward(self.decoder_unit_layer[0], latent)
        latent = one_input_forward(func.silu, latent)
        latent = one_input_forward(self.decoder_unit_layer[1], latent)
        latent = one_input_forward(self.decoder_unit_layer[2], latent)
        latent = one_input_forward(func.silu, latent)
        latent = one_input_forward(self.decoder_unit_layer[3], latent)
        latent = latent + one_input_forward(self.decoder_unit_layer[4], residue)

        return latent

class Token_Processing_Unit(nn.Module):
    def __init__(self, embed_dim, n_head):
        super().__init__()

        self.token_processing_unit_layer = nn.ModuleList([
            nn.LayerNorm(embed_dim),
            nn.MultiheadAttention(embed_dim, n_head, batch_first = True),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.Linear(4 * embed_dim, embed_dim)
        ])

    def forward(self, x, mask = None):
        residue = x
        x = self.token_processing_unit_layer[0](x)
        x, _ = self.token_processing_unit_layer[1](x, x, x, attn_mask = mask)
        x = x + residue
        
        residue = x
        x = self.token_processing_unit_layer[2](x)
        x = self.token_processing_unit_layer[3](x)
        x = x * func.sigmoid(1.702 * x)
        return self.token_processing_unit_layer[4](x) + residue

class VAE_Unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.VAE_unit_layer = nn.ModuleList([
            nn.GroupNorm(32, in_channels),
            nn.Conv2d(in_channels, out_channels, 3, padding = 1),
            nn.GroupNorm(32, out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding = 1),
            nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
        ])

    def forward(self, x):
        residue = x
        x = one_input_forward(self.VAE_unit_layer[0], (x))
        x = one_input_forward(func.silu, x)
        x = one_input_forward(self.VAE_unit_layer[1], x)
        x = one_input_forward(self.VAE_unit_layer[2], x)
        x = one_input_forward(func.silu, x)
        x = one_input_forward(self.VAE_unit_layer[3], x)
        residue = one_input_forward(self.VAE_unit_layer[4], residue)
        return x + residue
        
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.VAE_layer = nn.ModuleList([
            nn.Conv2d(3, 128, 3, padding = 1),
            VAE_Unit(128, 128),
            VAE_Unit(128, 128),
            nn.Conv2d(128, 128, 3, 2),
            VAE_Unit(128, 256),
            VAE_Unit(256, 256),
            nn.Conv2d(256, 256, 3, 2),
            VAE_Unit(256, 512),
            VAE_Unit(512, 512),
            nn.Conv2d(512, 512, 3, 2),
            VAE_Unit(512, 512),
            VAE_Unit(512, 512),
            VAE_Unit(512, 512),
            nn.GroupNorm(32, 512),
            nn.MultiheadAttention(512, 1, batch_first = True),
            VAE_Unit(512, 512),
            nn.GroupNorm(32, 512),
            nn.Conv2d(512, 32, 3, padding = 1),
            nn.Conv2d(32, 32, 1),
        ])

    def forward(self, x):
        for i in range(3):
            x = one_input_forward(self.VAE_layer[i * 3], x)
            x = self.VAE_layer[i * 3 + 1](x)
            x = self.VAE_layer[i * 3 + 2](x)
            x = func.pad(x, [0, 1, 0, 1])

        x = one_input_forward(self.VAE_layer[9], x)
        x = self.VAE_layer[10](x)
        x = self.VAE_layer[11](x)
        x = self.VAE_layer[12](x)

        residue = x
        x = one_input_forward(self.VAE_layer[13], x)
        h, w = x.shape[-2:]
        x = x.reshape(-1, h * w, 512)

        x = three_input_forward(self.VAE_layer[14], x, x, x)
        x = x.reshape(-1, 512, h, w) + residue


        x = self.VAE_layer[15](x)
        x = one_input_forward(self.VAE_layer[16], x)
        x = one_input_forward(func.silu, x)
        x = one_input_forward(self.VAE_layer[17], x)
        x = one_input_forward(self.VAE_layer[18], x)

        mean_tensor, log_variance_tensor = x.chunk(2, 1)
        std_tensor = log_variance_tensor.clamp(-30, 20).exp() ** 0.5

        return mean_tensor + std_tensor * torch.randn(mean_tensor.shape, device = self.device)
        
class Diffusion_Video_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.text_embedding_layer = nn.Embedding(50000, 768)

        self.text_processing_layer = nn.ModuleList(
            [Token_Processing_Unit(768, 12) for _ in range(12)] + [nn.LayerNorm(768)]
        )

        self.memory_latent_processing_layer = nn.ModuleList(
            [Token_Processing_Unit(64, 1) for _ in range(12)] + [nn.LayerNorm(64)]
        )

        self.a = torch.linspace(0.99, 0.97, 1000, device = self.device) ** 2
        self.A = self.a.cumprod(0)

        self.forward_diffusion_layer = nn.ModuleList([
            nn.Linear(320, 4 * 320),
            nn.Linear(4 * 320, 4 * 320),
            nn.Conv2d(16, 320, 3, padding = 1),
            Diffusion_Unit(320, 320),
            Diffusion_Unit(320, 320),
            nn.Conv2d(320, 320, 3, 2, 1),
            Diffusion_Unit(320, 640),
            Diffusion_Unit(640, 640),
            nn.Conv2d(640, 640, 3, 2, 1),
            Diffusion_Unit(640, 1280),
            Diffusion_Unit(1280, 1280),
            nn.Conv2d(1280, 1280, 3, 2, 1),
            Diffusion_First_Unit(1280, 1280),
            Diffusion_First_Unit(1280, 1280),
            Diffusion_Unit(1280, 1280),
            Diffusion_First_Unit(1280, 1280),
            Diffusion_First_Unit(2560, 1280),
            Diffusion_First_Unit(2560, 1280),
            Diffusion_First_Unit(2560, 1280),
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(1280, 1280, 3, padding = 1),
            Diffusion_First_Unit(2560, 1280),
            Diffusion_Second_Unit(1280),
            Diffusion_First_Unit(2560, 1280),
            Diffusion_Second_Unit(1280),
            Diffusion_First_Unit(1920, 1280),
            Diffusion_Second_Unit(1280),
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(1280, 1280, 3, padding = 1),
            Diffusion_First_Unit(1920, 640),
            Diffusion_Second_Unit(640),
            Diffusion_First_Unit(1280, 640),
            Diffusion_Second_Unit(640),
            Diffusion_First_Unit(960, 640),
            Diffusion_Second_Unit(640),
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(640, 640, 3, padding = 1),
            Diffusion_First_Unit(960, 320),
            Diffusion_Second_Unit(320),
            Diffusion_First_Unit(640, 320),
            Diffusion_Second_Unit(320),
            Diffusion_First_Unit(640, 320),
            Diffusion_Second_Unit(320),
            nn.GroupNorm(32, 320),
            nn.Conv2d(320, 16, 3, padding = 1),
        ])

        self.decode_layer = nn.ModuleList([
            nn.Conv2d(16, 16, 1),
            nn.Conv2d(16, 512, 3, padding = 1),
            Decoder_Unit(512, 512),
            nn.GroupNorm(32, 512),
            nn.MultiheadAttention(512, 1, batch_first = True),
            Decoder_Unit(512, 512),
            Decoder_Unit(512, 512),
            Decoder_Unit(512, 512),
            Decoder_Unit(512, 512),
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(512, 512, 3, padding = 1),
            Decoder_Unit(512, 512),
            Decoder_Unit(512, 512),
            Decoder_Unit(512, 512),
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(512, 512, 3, padding = 1),
            Decoder_Unit(512, 256),
            Decoder_Unit(256, 256),
            Decoder_Unit(256, 256),
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(256, 256, 3, padding = 1),
            Decoder_Unit(256, 128),
            Decoder_Unit(128, 128),
            Decoder_Unit(128, 128),
            nn.GroupNorm(32, 128),
            nn.Conv2d(128, 3, 3, padding = 1)
        ])

        self.latent_tokenize_layer = nn.Conv1d(1, 64, 4096, 4096)

        self.encode_layer = VAE()

        self.autoencoder_optimizer = optim.Adam(
            list(self.encode_layer.parameters()) + list(self.decode_layer.parameters()),
            1e-4
        )
        self.autoencoder_criterion = nn.MSELoss()
        self.stable_diffusion_optimizer = optim.Adam(
            list(self.text_embedding_layer.parameters()) + 
            list(self.text_processing_layer.parameters()) +
            list(self.forward_diffusion_layer.parameters()) +
            list(self.latent_tokenize_layer.parameters()) +
            list(self.memory_latent_processing_layer.parameters()),
            1e-4
        )
        self.stable_diffusion_criterion = nn.MSELoss()

    def decode(self, latent):
        latent = one_input_forward(self.decode_layer[0], latent)
        latent = one_input_forward(self.decode_layer[1], latent)
        latent = self.decode_layer[2](latent)

        residue = latent
        latent = one_input_forward(self.decode_layer[3], latent)
        h, w = latent.shape[-2:]
        latent = latent.reshape(-1, h * w, 512)
        latent = three_input_forward(self.decode_layer[4], latent, latent, latent)
        latent = latent.reshape(-1, 512, h, w) + residue

        for i in range(5, 25):
            if isinstance(self.decode_layer[i], Decoder_Unit):
                latent = self.decode_layer[i](latent)
            else:
                latent = one_input_forward(self.decode_layer[i], latent)

        latent = one_input_forward(func.silu, latent)
        return one_input_forward(self.decode_layer[25], latent)

    def text_processing(self, text_embedding):
        x = text_embedding
        mask = torch.full((1000, 1000), float('-inf'), device = self.device)
        mask.masked_fill_(torch.ones(1000, 1000, device = self.device).tril(0).bool(), 0)
        
        for i in range(12):
            x = self.text_processing_layer[i](x, mask)

        return self.text_processing_layer[12](x)
    
    def latent_attention(self, memory_latent):
        x = memory_latent
        
        for i in range(12):
            x = self.memory_latent_processing_layer[i](x)

        return self.memory_latent_processing_layer[12](x)

    def latent_processing(self, latent, context, time_embedding, memory_latent):
        if (type(memory_latent) == list):
            memory_latent = torch.cat(memory_latent, 1)
        memory_latent = memory_latent + 0.5 * positional_encoder((memory_latent.shape[1], 64), 50000).to(self.device)

        time_encoding = self.forward_diffusion_layer[0](time_embedding)
        time_encoding = func.silu(time_encoding)
        time_encoding = self.forward_diffusion_layer[1](time_encoding)

        memory_latent = self.latent_attention(memory_latent)

        S = []
        for i in range(2, 14):
            if type(self.forward_diffusion_layer[i]) == nn.Conv2d:
                latent = self.forward_diffusion_layer[i](latent)
            elif type(self.forward_diffusion_layer[i]) == Diffusion_Unit:
                latent, time_encoding = self.forward_diffusion_layer[i](latent, time_encoding, context, memory_latent)
            elif type(self.forward_diffusion_layer[i]) == Diffusion_First_Unit:
                latent, time_encoding = self.forward_diffusion_layer[i](latent, time_encoding)
            else:
                latent = self.forward_diffusion_layer[i](latent, context, memory_latent)
            S.append(latent)

        latent, time_encoding = self.forward_diffusion_layer[14](latent, time_encoding, context, memory_latent)
        latent, time_encoding = self.forward_diffusion_layer[15](latent, time_encoding)

        i = 16
        while i <= 42:
            latent = torch.cat((latent, S.pop()), 1)
            latent, time_encoding = self.forward_diffusion_layer[i](latent, time_encoding)
            i += 1

            if i != 17 and i != 18 and i != 19:
                latent = self.forward_diffusion_layer[i](latent, context, memory_latent)
                i += 1

            if i == 19 or i == 27 or i == 35:
                latent = self.forward_diffusion_layer[i](latent)
                latent = self.forward_diffusion_layer[i + 1](latent)
                i += 2

        latent = self.forward_diffusion_layer[43](latent)
        latent = func.silu(latent)
        predicted_noise = self.forward_diffusion_layer[44](latent)
        return predicted_noise

    # previous_latent = (23, 16, 64, 96) => (23, 1, 16 * 64 * 96) => (23, 64, 24) => (23, 24, 64)
    def latent_tokenize(self, previous_latent):
        return self.latent_tokenize_layer(previous_latent.reshape(previous_latent.shape[0], 1, -1)).permute(0, 2, 1)

    def infer(self, prompts, latent_shape, frames):
        batch_size = len(prompts)
        h, w = latent_shape

        video = []
        debug_information = []
        memory_latent = [torch.zeros(batch_size, h * w // 256, 64, device = self.device)]

        BPE_tokenizer = transformers.CLIPTokenizer("vocabulary.json", "merge.txt", clean_up_tokenization_spaces = True)
        token_sentences = torch.tensor(BPE_tokenizer.batch_encode_plus(
            prompts, padding = "max_length", max_length = 1000
        ).input_ids, device = self.device)

        with torch.no_grad():
            text_embedding = self.text_embedding_layer(token_sentences) + 0.5 * positional_encoder((1000, 768), 2000).to(self.device)
            context = self.text_processing(text_embedding)

            for _ in range(frames):
                latent = torch.randn(batch_size, 16, h, w, device = self.device)

                for t in range(980, 0, -20):
                    time_embedding = time_encoder(320, t).reshape(1, 320).to(self.device)
                    predicted_noise = self.latent_processing(latent, context, time_embedding, memory_latent)

                    At = self.A[t]
                    At_k = self.A[t - 20]

                    latent = \
                        At_k ** 0.5 * (1 - At / At_k) / (1 - At) * predicted_noise + \
                        (At / At_k) ** 0.5 * (1 - At_k) / (1 - At) * latent + \
                        ((1 - At_k) / (1 - At) * (1 - At / At_k)) ** 0.5 * torch.randn(latent.shape, device = self.device)
                
                memory_latent.append(self.latent_tokenize(latent))
                video.append(((self.decode(latent) + 1) * 255 / 2).to("cpu", dtype = torch.int32).clamp(0, 255))
                debug_information.append(self.decode(latent))

        return (video, debug_information)
    
    def one_step_train_auto_encoder(self, batch_frames, verbose = False):
        loss = self.autoencoder_criterion(self.decode(self.encode_layer(batch_frames)), batch_frames)
        if verbose:
            print("Autoencoder Loss = " + str(loss.item()))

        loss.backward()
        self.autoencoder_optimizer.step()
        self.autoencoder_optimizer.zero_grad()
        if verbose:
            print("Autoencoder Stepped")

        return loss.item()


    def one_step_train_stable_diffusion(self, memory_latent, prompt, verbose = False):
        _, frames, _, height, width = memory_latent.shape
        
        random_frame = random.randint(0, frames - 1)
        # (1, 16, 64, 96)
        chosen_latent = memory_latent[:, random_frame]

        random_time = random.randint(0, 999)
        time_embedding = time_encoder(320, random_time).reshape(1, 320).to(self.device)

        # shape (1, 16, 64, 96)
        added_noise = torch.randn(chosen_latent.shape, device = self.device)
        noise_latent = \
            self.A[random_time] ** 0.5 * chosen_latent + \
            (1 - self.A[random_time]) ** 0.5 * added_noise
        
        BPE_tokenizer = transformers.CLIPTokenizer("vocabulary.json", "merge.txt", clean_up_tokenization_spaces = True)
        token_sentences = torch.tensor(BPE_tokenizer.batch_encode_plus(
            prompt, padding = "max_length", max_length = 1000
        ).input_ids, device = self.device)

        # shape (1, 1000, 768)
        context = self.text_processing(
            self.text_embedding_layer(token_sentences) + 
            0.5 * positional_encoder((1000, 768), 2000).to(self.device)
        )


        # (1, 23, 16, 64, 96)
        previous_latent = torch.cat((
            torch.zeros(1, 1, 16, height // 8, width // 8, device = self.device), 
            memory_latent[:, :random_frame]
        ), 1)
        # (23, 24, 64)
        previous_latent = self.latent_tokenize(
            previous_latent.reshape(-1, 16, height // 8, width // 8)
        ).reshape(1, -1, 64)
        
        predicted_noise = self.latent_processing(noise_latent, context, time_embedding, previous_latent)
        loss = self.criterion(predicted_noise, added_noise)
        if verbose:
            print("Stable Diffusion Loss = " + str(loss.item()))

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        if verbose:
            print("Stable Diffusion Stepped")

        return loss.item()

    def train_auto_encoder(self, time_step):
        resolution = time_step % 6
        if resolution == 0:
            # change this shit to 512 384
            resolution = [256, 256]
        elif resolution == 1:
            resolution = [768, 512]
        elif resolution == 2:
            resolution = [1024, 640]
        elif resolution == 3:
            resolution = [1408, 896]
        elif resolution == 4:
            resolution = [1664, 1024]
        elif resolution == 5:
            resolution = [1920, 1280]

        batch_video = []
        losses = []

        for f in os.listdir("videos"):
            curent_video = []
            video_generator = cv.VideoCapture(os.path.join("videos", f))
            for _ in range(64):
                curent_video.append(torch.from_numpy(cv.resize(video_generator.read()[1], resolution)).permute(2, 0, 1))
            video_generator.release()
            batch_video.append(torch.stack(curent_video))
  
        # (128, 3, 512, 768)
        batch_video = torch.cat(batch_video).to(self.device) / 255. * 2 - 1

        for _ in range(100):
            random_index = random.randint(0, 31)
            batch_frames = batch_video[random_index * 4:random_index * 4 + 4]
            loss = self.one_step_train_auto_encoder(batch_frames, True)
            losses.append(loss)
            torch.cuda.empty_cache()
            a = None * 2

        return sum(losses) / len(losses)
    
    def train_stable_diffusion(self, time_step):
        _, frames = configuration_at_time_step(time_step)
        resolution = time_step % 6
        if resolution == 0:
            resolution = [512, 384]
        elif resolution == 1:
            resolution = [768, 512]
        elif resolution == 2:
            resolution = [1024, 640]
        elif resolution == 3:
            resolution = [1408, 896]
        elif resolution == 4:
            resolution = [1664, 1024]
        elif resolution == 5:
            resolution = [1920, 1280]

        losses = []
        video = []
        prompt = []
        video_generator = cv.VideoCapture("videos/video0.mp4")
        for _ in range(frames):
            video.append(torch.from_numpy(cv.resize(video_generator.read()[1], resolution)).permute(2, 0, 1))
        video_generator.release()
        # (64, 3, 512, 768)
        video = (torch.stack(video) / 255. * 2 - 1).to(self.device)
        with open("videos/description0.txt") as df:
            prompt.append(df.read())

        memory_latent = []
        with torch.no_grad():
            for i in range(frames // 4):
                memory_latent.append(self.encode_layer(video[i:i + 4]))

        # (1, 64, 16, 64, 96)
        memory_latent = torch.cat(memory_latent).unsqueeze(0)

        for _ in range(100):
            loss = self.one_step_train_stable_diffusion(memory_latent, prompt, True)
            losses.append(loss)
            torch.cuda.empty_cache()

        return sum(losses) / len(losses)
    
    def save(self):
        torch.save({
            "params" : self.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }, "model.ckpt")

        print("Model has been saved successfully.")

    def load(self):
        model = torch.load("model.ckpt")
        self.load_state_dict(model["params"])
        self.optimizer.load_state_dict(model["optimizer"])

        print("Model has been loaded successfully.")
