import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv → ReLU)×2, padding=1 to preserve spatial size."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then DoubleConv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then DoubleConv. Uses ConvTranspose2d for upsampling."""
    def __init__(self,x1_ch, x2_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(x1_ch, x1_ch//2, kernel_size=2, stride=2)
        self.conv = DoubleConv((x1_ch // 2) + x2_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # pad if needed (due to rounding)  
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX//2, diffX - diffX//2,
                        diffY//2, diffY - diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNetColor(nn.Module):
    """
    U-Net with color conditioning.
    Based on Ronneberger et al. (2015): contracting + expanding path with skip connections.
    Color is embedded and concatenated as extra channels at input.
    """
    def __init__(self, n_colors, embed_dim=16):
        super().__init__()
        self.color_emb = nn.Embedding(n_colors, embed_dim)

        # After concatenating embed_dim channels to the 1-channel mask
        self.inc  = DoubleConv(1 + embed_dim, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(x1_ch=512, x2_ch=512, out_ch=256)
        self.up2 = Up(x1_ch=256, x2_ch=256, out_ch=128)
        self.up3 = Up(x1_ch=128, x2_ch=128, out_ch= 64)
        self.up4 = Up(x1_ch= 64, x2_ch= 64, out_ch= 64)

        self.outc  = nn.Conv2d(64, 3, kernel_size=1)  # 3 output channels (RGB)

    def forward(self, x, color_idx):
        # x: [B,1,H,W], color_idx: [B]
        b, _, h, w = x.shape
        e = self.color_emb(color_idx)           # [B,embed_dim]
        e = e.view(b, -1, 1, 1).expand(b, -1, h, w)  # [B,embed_dim,H,W]
        x = torch.cat([x, e], dim=1)            # [B,1+embed_dim,H,W]

        x1 = self.inc(x)    # → [B,64,H,W]
        x2 = self.down1(x1) # → [B,128,H/2,W/2]
        x3 = self.down2(x2) # → [B,256,H/4,W/4]
        x4 = self.down3(x3) # → [B,512,H/8,W/8]
        x5 = self.down4(x4) # → [B,512,H/16,W/16]

        x  = self.up1(x5, x4)  # →
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        out = torch.sigmoid(self.outc(x))  # in [0,1] for RGB
        return out
