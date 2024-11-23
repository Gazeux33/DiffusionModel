import os
import torchvision
from PIL import Image

def setup_logging(args):
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.join(args.checkpoints_dir, args.run_name), exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, args.run_name), exist_ok=True)



def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)