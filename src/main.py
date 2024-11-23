from src.train.trainer import Trainer
import logging
import torch

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.checkpoints_dir = "checkpoints"
    args.results_dir = "results"
    args.load_model = None
    args.load_model = "checkpoints/DDPM_Uncondtional_cat"
    args.run_name = "DDPM_Uncondtional_cat"
    args.epochs = 500
    args.batch_size = 5
    args.image_size = 64
    args.dataset_path = r"data/cats"
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.lr = 3e-4
    args.save_model_freq = 5
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    launch()