import torch
from tqdm import tqdm

from src.model.diffusion import Diffusion
from src.model.unet import UNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet = UNet().to(device)
unet.load_state_dict(torch.load("../checkpoints/DDPM_Uncondtional_cat/ckpt_395.pt", weights_only=True))

seed = torch.randint(0, 100000, (1,)).item()
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


diffusion = Diffusion(noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device=device)
diffusion.sample_with_step(unet, 1,"results/image_gif3")

