import os
import torch


def save_checkpoint(model, optimizer, epoch, checkpoint_path, scheduler=None, best=False):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, checkpoint_path)
    status = "BEST " if best else ""
    print(f"{status}Checkpoint saved at '{checkpoint_path}' (epoch {epoch})")


def load_checkpoint(model, optimizer=None, checkpoint_path=None, scheduler=None, device='cuda'):
    if checkpoint_path is None or not os.path.isfile(checkpoint_path):
        print(f"Checkpoint path {checkpoint_path} not found")
        return 0

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint.get("epoch", 0) + 1
    print(f"Checkpoint loaded at '{checkpoint_path}' (epoch {start_epoch})")
    return start_epoch