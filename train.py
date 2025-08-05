import os
import argparse
import torch
import wandb
from torch.utils.data import DataLoader
from torch import nn, optim

from data_loader import PolygonColorDataset
from model import UNetColor

def train(args):
    # 1) Init W&B
    wandb.init(project="colored-polygon-unet", config=vars(args))
    cfg = wandb.config

    # 2) Prepare data
    COLORS = [
    "cyan", "purple", "magenta", "green",
    "red",  "blue",   "yellow",  "orange"
] # extend as needed
    train_ds = PolygonColorDataset("dataset/training/", COLORS, img_size=(128,128))
    val_ds   = PolygonColorDataset("dataset/validation/", COLORS, img_size=(128,128))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=4)

    # 3) Model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetColor(n_colors=len(COLORS), embed_dim=cfg.embed_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    # 4) Training loop
    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0
        for inp, cidx, tgt in train_loader:
            inp, cidx, tgt = inp.to(device), cidx.to(device), tgt.to(device)
            pred = model(inp, cidx)
            loss = criterion(pred, tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 5) Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inp, cidx, tgt in val_loader:
                inp, cidx, tgt = inp.to(device), cidx.to(device), tgt.to(device)
                pred = model(inp, cidx)
                val_loss += criterion(pred, tgt).item()

        # 6) Log & checkpoint
        avg_tr = train_loss / len(train_loader)
        avg_va = val_loss   / len(val_loader)
        wandb.log({"train_loss": avg_tr, "val_loss": avg_va}, step=epoch)
        ckpt_path = os.path.join("checkpoints", f"epoch{epoch:03d}.pth")
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)

        print(f"Epoch {epoch}/{cfg.epochs} → train: {avg_tr:.4f}, val: {avg_va:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",    type=int,   default=30)
    p.add_argument("--batch_size",type=int,   default=16)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--embed_dim", type=int,   default=16)
    args = p.parse_args()
    train(args)


