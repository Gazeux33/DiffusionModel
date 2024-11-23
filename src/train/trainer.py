import torch.optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import logging

from src.model.unet import UNet
from src.model.diffusion import Diffusion
from src.utils import setup_logging,save_images
from src.data.get_data import get_data



class Trainer:
    def __init__(self,args):
        setup_logging(args)
        self.args = args
        self.device = args.device
        self.dataloader = get_data(args)
        self.model = UNet().to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)
        self.mse = nn.MSELoss()
        self.diffusion = Diffusion(img_size=args.image_size, device=self.device)
        self.logger = SummaryWriter(os.path.join("runs", args.run_name))
        self.l = len(self.dataloader)
        self.epochs = args.epochs
        self.begin_epoch = 0
        self.results_dir = args.results_dir
        self.run_name = args.run_name
        self.save_model_freq = args.save_model_freq
        self.checkpoints_dir = args.checkpoints_dir
        self.load_dir = args.load_model

        if self.load_dir is not None:
            self.load_model()
        
    def train(self):
        for epoch in range(self.begin_epoch,self.epochs):
            logging.info(f"Starting epoch {epoch}:")
            pbar = tqdm(self.dataloader)
            for i, batch in enumerate(pbar):
                if len(batch) == 2:
                    images, _ = batch
                else:
                    images = batch
                images = images.to(self.device)
                t = self.diffusion.sample_timesteps(images.shape[0]).to(self.device)
                x_t, noise = self.diffusion.noise_images(images, t)
                predicted_noise = self.model(x_t, t)
                loss = self.mse(noise, predicted_noise)
    
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
                pbar.set_postfix(MSE=loss.item())
                self.logger.add_scalar("MSE", loss.item(), global_step=epoch * self.l + i)
    
            sampled_images = self.diffusion.sample(self.model, n=images.shape[0])
            save_images(sampled_images, os.path.join(self.results_dir, self.run_name, f"{epoch}.jpg"))
            if epoch% self.save_model_freq == 0:
                torch.save(self.model.state_dict(), str(os.path.join(self.checkpoints_dir, self.run_name, f"ckpt_{epoch}.pt")))
    
    def load_model(self):
        assert os.path.exists(self.load_dir) , f"{self.load_dir} doesn't exist"
        list_save = os.listdir(self.load_dir)
        assert list_save , f"{self.load_dir} is empty"
        list_int = [int(l[l.index("_")+1:l.index(".")]) for l in list_save]
        list_int.sort()
        last = list_int[-1]
        self.model.load_state_dict(torch.load(os.path.join(self.args.load_model,f"ckpt_{last}.pt"), weights_only=True))
        self.begin_epoch = last +1
        logging.info(f"starting at epochs {self.begin_epoch}")
        

    

        

    
            

    
    
