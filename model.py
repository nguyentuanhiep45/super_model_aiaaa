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
            nn.GroupNorm(32, 320),
            nn.Conv2d(320, 320, 3, padding = 1),
            nn.Linear(4 * 320, 320),
            nn.GroupNorm(32, 320),
            nn.Conv2d(320, 320, 3, padding = 1),
            nn.GroupNorm(32, 320),
            nn.Conv2d(320, 320, 1),
            nn.LayerNorm(320),
            nn.MultiheadAttention(320, 8, batch_first = True),
            nn.LayerNorm(320),
            nn.Linear(320, 320 * 8),
            nn.Linear(4 * 320, 320),
            nn.Conv2d(320, 320, 1)
        ])

    def forward(self, x):
        return x
    
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
        latent_ = self.forward_diffusion_layer[2](latent)
        S.append(latent_)

        residue = latent_
        latent_ = self.forward_diffusion_layer[3](latent_)
        latent_ = func.silu(latent_)
        latent_ = self.forward_diffusion_layer[4](latent_)

        time_encoding = func.silu(time_encoding)
        time_ = self.forward_diffusion_layer[5](time_encoding).reshape(1, 320, 1, 1)
        latent_ += time_
        latent_ = self.forward_diffusion_layer[6](latent_)
        latent_ = func.silu(latent_)
        latent_ = self.forward_diffusion_layer[7](latent_)
        latent_ += residue

        residue_long = latent_
        latent_ = self.forward_diffusion_layer[8](latent_)
        latent_ = self.forward_diffusion_layer[9](latent_)
        latent_ = latent_.reshape(-1, 64 * 96, 320)
        residue_short = latent_
        latent_ = self.forward_diffusion_layer[10](latent_)
        latent_, _ = self.forward_diffusion_layer[11](latent_, latent_, latent_)
        latent_ += residue_short

        residue_short = latent_
        latent_ = self.forward_diffusion_layer[12](latent_)
        latent_, gate = self.forward_diffusion_layer[13](latent_).chunk(2, -1)
        gate = func.gelu(gate)
        latent_ *= gate
        latent_ = self.forward_diffusion_layer[14](latent_)
        latent_ += residue_short
        latent_ = latent_.reshape(-1, 320, 64, 96)
        latent_ = self.forward_diffusion_layer[15](latent_)
        latent_ += residue_long
        S.append(latent_)

        print(latent_.shape)
        exit()

    def infer(self, batch_input_text):
        BPE_tokenizer = transformers.CLIPTokenizer("vocabulary.json", "merge.txt", clean_up_tokenization_spaces = True)
        batch_token_sentence = torch.tensor(BPE_tokenizer.batch_encode_plus(
            batch_input_text, padding = "max_length", max_length = 1000
        ).input_ids, device = self.device)

        token_embedding = self.embedding(batch_token_sentence) + self.positional_encoding
        context_tensor = self.prompt_attention(token_embedding)

        latent = torch.randn(len(batch_input_text), 16, 64, 96, device = self.device)

        for time_step in range(980, -20, -20):
            time_embedding = time_encoder(320, time_step).reshape(1, 320).to(self.device)
            post_latent = self.latent_processing(latent, context_tensor, time_embedding)

        return post_latent








    