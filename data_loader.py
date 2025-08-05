import os, json
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class PolygonColorDataset(Dataset):
    def __init__(self, root_dir, colour_list, img_size=(128,128)):
        super().__init__()
        self.root = root_dir
        with open(os.path.join(root_dir, "data.json"), "r") as f:
            self.examples = json.load(f)

        # use exactly the same spelling as in your JSON
        self.colour2idx = {c: i for i, c in enumerate(colour_list)}

        self.transform_inp = T.Compose([
            T.Grayscale(),           # ensure 1 channel
            T.Resize(img_size),
            T.ToTensor(),
        ])
        self.transform_out = T.Compose([
            T.Resize(img_size),
            T.ToTensor(),            # 3×H×W
        ])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        rec = self.examples[idx]

        # **UPDATED** to match your JSON schema:
        inp_path  = os.path.join(self.root,"inputs", rec["input_polygon"])
        out_path  = os.path.join(self.root,"outputs", rec["output_image"])
        colour    = rec["colour"]  # note British spelling

        inp = Image.open(inp_path).convert("RGB")
        out = Image.open(out_path).convert("RGB")

        inp = self.transform_inp(inp)   # [1,H,W]
        out = self.transform_out(out)   # [3,H,W]

        color_idx = self.colour2idx[colour]
        return inp, torch.tensor(color_idx, dtype=torch.long), out
