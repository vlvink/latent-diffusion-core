import argparse
import warnings

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.applications.train_vae import train_vae
from src.models.autoencoder import VAE
from src.data.vae_dataset import VaeDataset

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--epochs", required=True, type=int, help="number of epochs"
    )
    parser.add_argument(
        "-b", "--batch_size", required=True, type=int, help="batch size"
    )
    args = vars(parser.parse_args())

    print("[INFO] Initializing VAE...\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = VAE()

    print("[INFO] Prepating data...\n")
    pretrain_dataset = VaeDataset()
    pretrain_dataloader = DataLoader(pretrain_dataset, batch_size=args["batch_size"], shuffle=True)

    eval_dataset = VaeDataset(mode="val")
    eval_dataloader = DataLoader(eval_dataset, batch_size=args["batch_size"], shuffle=False)

    optimizer = AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-4 / 1e-4,
        end_factor=1.0,
        total_iters=int(args["epochs"] * 0.1),
    )

    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args["epochs"] - int(args["epochs"] * 0.1),
        eta_min=1e-6
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[int(args["epochs"] * 0.1)]
    )

    print("[INFO] Starting training")
    train_vae(
        model=model,
        device=device,
        train_loader=pretrain_dataloader,
        eval_loader=eval_dataloader,
        optimizer=optimizer,
        num_epochs=args["epochs"],
        scheduler=scheduler,
        accumulation_steps=1,
        logging_steps=100,
        checkpoint_path="models/checkpoints_vae",
        beta_start=0.00025,
        beta_end=0.01
    )