# Correa (last-name, first-name)
# 100x_xxx_xxx
# 2026_03_30
# Assignment_04

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_lfw_people
import numpy as np
from vae_model import VAE, vae_loss

LATENT_DIM = 6
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-3
IMAGE_SIZE = 64
MODEL_PATH = "vae_model.pth"


def load_lfw_data():
    print("Downloading/loading LFW dataset...", flush=True)
    lfw = fetch_lfw_people(min_faces_per_person=1, resize=1.0, color=True)
    images = lfw.images  # (N, H, W, 3), values in [0, 255]
    print(f"Loaded {len(images)} images, original shape: {images.shape[1:]}", flush=True)

    # Normalize to [0, 1]
    images = images / 255.0 if images.max() > 1.0 else images

    # Convert to (N, C, H, W) tensor and resize with PyTorch (much faster than skimage loop)
    tensor = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)
    tensor = F.interpolate(tensor, size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)
    print(f"Resized to {IMAGE_SIZE}x{IMAGE_SIZE}", flush=True)
    return tensor


def train():
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Using device: {device}", flush=True)

    data = load_lfw_data()
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = VAE(latent_dim=LATENT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\nTraining VAE for {EPOCHS} epochs...", flush=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for (batch,) in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, log_var = model(batch)
            loss = vae_loss(recon, batch, mu, log_var)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(data)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{EPOCHS}] - Avg Loss: {avg_loss:.4f}", flush=True)

    # Save model (move to CPU for portability)
    model = model.cpu()
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}", flush=True)


if __name__ == "__main__":
    train()
