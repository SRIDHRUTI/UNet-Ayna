# Colored-Polygon U-Net

**A from-scratch PyTorch U-Net that takes a binary polygon mask + a color name and produces an RGB image of the polygon filled with the specified color. Includes data loading, training, inference notebook, and a W&B sweep for hyper-parameter search.**

## Requirements & Setup

1. **Clone the repo** and `cd` into it.  
2. **Install dependencies**:
   ```bash
   pip install torch torchvision wandb pillow matplotlib scikit-learn


Training
Run the training script to start a W&B-tracked experiment and save model checkpoints:
wandb login
python train.py \
  --epochs 30 \
  --batch_size 16 \
  --lr 1e-3 \
  --embed_dim 16

Key Hyperparameters
--epochs	30	Number of training epochs
--batch_size	16	Batch size (GPU memory vs. gradient noise trade-off)
--lr	1e-3	Learning rate
--embed_dim	16	Dimension of the color embedding vector
