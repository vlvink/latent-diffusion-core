import torch
import torch.nn.functional as F
import lpips
from tqdm import tqdm

from .checkpoints import save_checkpoint


def vae_loss(recon_x, x, mu, logvar, beta=0.001):
    mse = F.mse_loss(recon_x, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + beta*kl_loss, (mse, kl_loss)


def train_vae(
        model, device, train_loader, eval_loader, optimizer, num_epochs, scheduler=None, accumulation_steps=1,
        logging_steps=100, checkpoint_path="models/checkpoints_vae",
        beta_start=0.00025, beta_end=0.01
):
    lpips_score = lpips.LPIPS(net='vgg').to(device)
    model.to(device)

    best_lpips = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss, running_mse, running_kl = 0.0, 0.0, 0.0

        beta = beta_start + (beta_end - beta_start) * (epoch / (num_epochs - 1))

        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")
        for i, images in loop:
            images = images.to(device)

            recon_x, encoded = model(images)
            mean, log_variance = torch.chunk(encoded, 2, dim=1)

            loss, (mse, kl_loss) = vae_loss(recon_x, images, mean, log_variance, beta)
            loss = loss / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * accumulation_steps
            running_mse += mse.item()
            running_kl += kl_loss.item()

            if (i + 1) % logging_steps == 0:
                with torch.no_grad():
                    lpips_value = lpips_score(images, recon_x).mean().item()
                loop.set_postfix({
                    "Loss": f"{running_loss / logging_steps:.4f}",
                    "MSE": f"{running_mse / logging_steps:.4f}",
                    "KL": f"{running_kl / logging_steps:.4f}",
                    "LPIPS": f"{lpips_value:.4f}",
                    "β": f"{beta:.6f}"
                })
                running_loss, running_mse, running_kl = 0.0, 0.0, 0.0

        if scheduler:
            scheduler.step()

        epoch_ckpt_path = f"{checkpoint_path}/chkpt_vae_epoch_{epoch}.pth"
        save_checkpoint(model, optimizer, epoch, epoch_ckpt_path, scheduler)

        model.eval()
        lpips_scores = []
        with torch.no_grad():
            for images in tqdm(eval_loader, desc="Validation"):
                images = images.to(device)
                recon_x, _ = model(images)
                lpips_value = lpips_score(images, recon_x).mean().item()
                lpips_scores.append(lpips_value)

        avg_lpips_score = sum(lpips_scores) / len(lpips_scores)
        print(f"Epoch {epoch} Validation LPIPS: {avg_lpips_score:.4f}")

        if avg_lpips_score < best_lpips:
            best_lpips = avg_lpips_score
            best_ckpt_path = f"{checkpoint_path}/best_vae.pth"
            save_checkpoint(model, optimizer, epoch, best_ckpt_path, scheduler, best=True)

    print(f"\nTraining finished. Best Validation LPIPS: {best_lpips:.4f}")
