import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from rich.logging import RichHandler
from torchvision.utils import save_image


class Logger:
    def __init__(self):
        self.name = "Latent-Inv"

    def initLogger(self):
        __logger = logging.getLogger(self.name)

        FORMAT = f"[{self.name}] >> %(message)s"
        handler = RichHandler()
        handler.setFormatter(logging.Formatter(FORMAT))

        __logger.addHandler(handler)

        __logger.setLevel(logging.INFO)

        return __logger

def make_gif(input_path: Path, save_path: Path) -> None:
    files = sorted(input_path.glob('*.png'))
    frames = []

    for f in files:
        frames.append(Image.open(f).convert('RGB'))

    frame_one = frames[0]
    frame_one.save(save_path, format="GIF", append_images=frames,
                   save_all=True, duration=100, loop=0)

def save_floats(data: list, save_path: Path) -> None:
    with open(save_path, 'w') as f:
        for item in data:
            f.write(f"{item}\n")

def create_workdir(workdir: Path) -> None:
    workdir.joinpath('result').mkdir(parents=True, exist_ok=True)

def set_seed(seed: int):
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)