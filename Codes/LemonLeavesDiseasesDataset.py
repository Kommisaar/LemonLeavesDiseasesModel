from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2


class LemonLeavesDiseasesDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.transform = transform
        self.images_data = data
        self.images_label = target
        self.non_train_transformer = v2.Compose([
            v2.Resize(size=(480, 480)),
            v2.PILToTensor(),
            v2.ToDtype(dtype=torch.float32)
        ])

    def __len__(self) -> int:
        return len(self.images_data)

    def __getitem__(self, idx) -> tuple[Image.Image, str]:
        image_data = Image.open(self.images_data[idx])
        label_data = self.images_label[idx]

        if self.transform is not None:
            image_data = self.transform(image_data)
        else:
            image_data = self.non_train_transformer(image_data)

        return image_data, label_data


def lemon_leaves_load_data(root_dir: Path) -> Tuple[list, np.array]:
    images_data = []
    images_label = []
    labels_dict = {image_label: index for index, image_label in enumerate(root_dir.iterdir())}

    for image_dir in root_dir.iterdir():
        for file in image_dir.iterdir():
            images_data.append(file)
            images_label.append(labels_dict[image_dir])

    return images_data, np.array(images_label)
