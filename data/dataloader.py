from PIL import Image
from torch.utils.data import Dataset
import os


class ImagesDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.imgs = os.listdir(img_dir)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)