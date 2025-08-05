# sweep.py
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_loader import PolygonColorDataset
from model import UNetColor

# 1) Define your sweep config
sweep_config = {
    "method": "grid",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "lr":           {"values": [1e-2, 1e-3, 1e-4]},
        "weight_decay": {"values": [0.0, 1e-5, 1e-4]},
        "batch_size":   {"values": [8, 16, 32]},
        "embed_dim":    {"values": [8, 16, 32]},
    }
}

# 2) Initialize sweep
sweep_id = wandb.sweep(sweep_config, project="colored-polygon-unet-sweeps")

def sweep_train():
    # 3) Initialize a run
    wandb.init()
    cfg = wandb.config

    # 4) Data loaders
    COLOURS = ["cyan","purple","magenta","green","red","blue","yellow","orange"]
    train_ds = PolygonColorDataset("dataset/training", COLOURS, img_size=(128,128))
    val_ds   = PolygonColorDataset("dataset/validation", COLOURS, img_size=(128,128))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size,
                              shuffle=False, num_workers=4)

    # 5) Model & optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetColor(n_colors=len(COLOURS), embed_dim=cfg.embed_dim).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()

    # 6) Training loop (fewer epochs for sweep)
    for epoch in range(5):
        model.train()
        for inp, cidx, tgt in train_loader:
            inp, cidx, tgt = inp.to(device), cidx.to(device), tgt.to(device)
            pred = model(inp, cidx)
            loss = criterion(pred, tgt)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

    # 7) Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inp, cidx, tgt in val_loader:
            inp, cidx, tgt = inp.to(device), cidx.to(device), tgt.to(device)
            val_loss += criterion(model(inp, cidx), tgt).item()
    val_loss /= len(val_loader)

    # 8) Log final val_loss
    wandb.log({"val_loss": val_loss})

# 9) Launch agents
if __name__ == "__main__":
    wandb.agent(sweep_id, function=sweep_train, count= len(sweep_config["parameters"]["lr"]) *
                                                        len(sweep_config["parameters"]["weight_decay"]) *
                                                        len(sweep_config["parameters"]["batch_size"]) *
                                                        len(sweep_config["parameters"]["embed_dim"]))
