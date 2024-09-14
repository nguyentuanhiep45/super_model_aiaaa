from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import torch
from torch import nn
from torch.nn import functional as func
import transformers

def exist_model_in_drive():
    auth = GoogleAuth()
    drive = GoogleDrive(auth)

    auth.LoadCredentialsFile("drive_token.json")
    if auth.access_token_expired:
        auth.Refresh()

    drive_files = drive.ListFile({
        "q": "'1r_oDc5Wm7rYqAPvcdRWsHKzjphgu__C3' in parents and trashed=false"
    }).GetList()

    for file in drive_files:
        if file["title"] == "model.ckpt": return True
    
    return False

def positional_encoder(shape):
    m = 2000 ** (2 / shape[1])
    temp = torch.arange(0, shape[0], dtype = torch.float32).reshape(shape[0], 1) @ \
           (m ** -torch.arange(0, shape[1] // 2)).unsqueeze(0)
    
    return torch.cat((torch.sin(temp), torch.cos(temp)), 1)

def time_encoder(size, time_step):
    m = 2000 ** (2 / size)
    temp = m ** -torch.arange(0, size // 2) * time_step
    return torch.cat((torch.sin(temp), torch.cos(temp)), 0)

class Diffusion_Sub_Unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.diffusion_unit_layer_1 = nn.ModuleList([
            nn.GroupNorm(32, in_channels),
            nn.Conv2d(in_channels, out_channels, 3, padding = 1),
            nn.Linear(4 * 320, out_channels),
            nn.GroupNorm(32, out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding = 1),
            nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
        ])

    def forward(self, latent, time_encoding):
        residue = latent
        latent = self.diffusion_unit_layer_1[0](latent)
        latent = func.silu(latent)
        latent = self.diffusion_unit_layer_1[1](latent)

        time_encoding = func.silu(time_encoding)
        latent = latent + self.diffusion_unit_layer_1[2](time_encoding).reshape(1, self.out_channels, 1, 1)
        latent = self.diffusion_unit_layer_1[3](latent)
        latent = func.silu(latent)
        latent = self.diffusion_unit_layer_1[4](latent) + self.diffusion_unit_layer_1[5](residue)

        return (latent, time_encoding)

class Diffusion_Sub_Unit_2(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.out_channels = out_channels

        self.diffusion_unit_layer_2 = nn.ModuleList([
            nn.GroupNorm(32, out_channels),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.LayerNorm(out_channels),
            nn.MultiheadAttention(out_channels, 8, batch_first = True),
            nn.LayerNorm(out_channels),
            nn.Linear(out_channels, out_channels * 8),
            nn.Linear(4 * out_channels, out_channels),
            nn.Conv2d(out_channels, out_channels, 1)
        ])

    def forward(self, latent, time_encoding):
        residue_long = latent
        latent = self.diffusion_unit_layer_2[0](latent)
        latent = self.diffusion_unit_layer_2[1](latent)
        h, w = latent.shape[-2:]
        latent = latent.reshape(-1, h * w, self.out_channels)
        residue_short = latent
        latent = self.diffusion_unit_layer_2[2](latent)
        latent = self.diffusion_unit_layer_2[3](latent, latent, latent)[0] + residue_short

        residue_short = latent
        latent = self.diffusion_unit_layer_2[4](latent)
        latent, gate = self.diffusion_unit_layer_2[5](latent).chunk(2, -1)
        latent = latent * func.gelu(gate)
        latent = self.diffusion_unit_layer_2[6](latent) + residue_short
        latent = latent.reshape(-1, self.out_channels, h, w)
        latent = self.diffusion_unit_layer_2[7](latent) + residue_long

        return (latent, time_encoding)

class Diffusion_Unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.diffusion_unit_layer = nn.ModuleList([
            Diffusion_Sub_Unit(in_channels, out_channels),
            Diffusion_Sub_Unit_2(out_channels)
        ])

    def forward(self, latent, time_encoding):
        latent, time_encoding = self.diffusion_unit_layer[0](latent, time_encoding)
        return self.diffusion_unit_layer[1](latent, time_encoding)


class Diffusion_Video_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding = nn.Embedding(50000, 768)
        self.positional_encoding = positional_encoder((1000, 768)).to(self.device)

        self.text_processing_layer = nn.ModuleList([
            nn.LayerNorm(768),
            nn.MultiheadAttention(768, 12, batch_first = True),
            nn.LayerNorm(768),
            nn.Linear(768, 4 * 768),
            nn.Linear(4 * 768, 768),
            nn.LayerNorm(768)
        ])

        self.a = torch.linspace(0.99, 0.97, 1000) ** 2
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
            Diffusion_Sub_Unit(1280, 1280),
            Diffusion_Sub_Unit(1280, 1280),
            Diffusion_Unit(1280, 1280),
            Diffusion_Sub_Unit(1280, 1280),
            Diffusion_Sub_Unit(2560, 1280),
            Diffusion_Sub_Unit(2560, 1280),
            Diffusion_Sub_Unit(2560, 1280),
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(1280, 1280, 3, padding = 1),
            Diffusion_Sub_Unit(2560, 1280),
            Diffusion_Sub_Unit_2(1280)
        ])

    def prompt_attention(self, token_embedding):
        x = token_embedding
        mask = torch.full((1000, 1000), float('-inf'), device = self.device)
        mask.masked_fill_(torch.ones(1000, 1000, device = self.device).tril(0).bool(), 0)
        
        # Mark : chinh lai thanh 12
        for _ in range(1):
            residue = x
            x = self.text_processing_layer[0](x)
            x, _ = self.text_processing_layer[1](x, x, x, attn_mask = mask)
            x += residue
            
            residue = x
            x = self.text_processing_layer[2](x)
            x = self.text_processing_layer[3](x)
            x *= func.sigmoid(1.702 * x)
            x = self.text_processing_layer[4](x)
            x += residue

        return self.text_processing_layer[5](x)

    def latent_processing(self, latent, context_tensor, time_embedding):
        time_encoding = self.forward_diffusion_layer[0](time_embedding)
        time_encoding = func.silu(time_encoding)
        time_encoding = self.forward_diffusion_layer[1](time_encoding)

        S = []
        for i in range(2, 14):
            if type(self.forward_diffusion_layer[i]) == nn.Conv2d:
                latent = self.forward_diffusion_layer[i](latent)
            else:
                latent, time_encoding = self.forward_diffusion_layer[i](latent, time_encoding)
            S.append(latent)

        latent, time_encoding = self.forward_diffusion_layer[14](latent, time_encoding)
        latent, time_encoding = self.forward_diffusion_layer[15](latent, time_encoding)

        latent = torch.cat((latent, S.pop()), 1)
        latent, time_encoding = self.forward_diffusion_layer[16](latent, time_encoding)
        latent = torch.cat((latent, S.pop()), 1)
        latent, time_encoding = self.forward_diffusion_layer[17](latent, time_encoding)

        latent = torch.cat((latent, S.pop()), 1)
        latent, time_encoding = self.forward_diffusion_layer[18](latent, time_encoding)
        latent = self.forward_diffusion_layer[19](latent)
        latent = self.forward_diffusion_layer[20](latent)

        latent = torch.cat((latent, S.pop()), 1)
        latent, time_encoding = self.forward_diffusion_layer[21](latent, time_encoding)
        latent, time_encoding = self.forward_diffusion_layer[22](latent, time_encoding)

        print(latent.shape)
        print(time_encoding.shape)
        exit()
        

    def infer(self, batch_input_text, latent_shape):
        BPE_tokenizer = transformers.CLIPTokenizer("vocabulary.json", "merge.txt", clean_up_tokenization_spaces = True)
        batch_token_sentence = torch.tensor(BPE_tokenizer.batch_encode_plus(
            batch_input_text, padding = "max_length", max_length = 1000
        ).input_ids, device = self.device)

        token_embedding = self.embedding(batch_token_sentence) + self.positional_encoding
        context_tensor = self.prompt_attention(token_embedding)

        h, w = latent_shape
        latent = torch.randn(len(batch_input_text), 16, h, w, device = self.device)

        for time_step in range(980, -20, -20):
            time_embedding = time_encoder(320, time_step).reshape(1, 320).to(self.device)
            post_latent = self.latent_processing(latent, context_tensor, time_embedding)

        return post_latent








    