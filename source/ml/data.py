import torch
from torch.utils.data import Dataset

from datasets import load_dataset

import numpy as np


class ImageDataset(Dataset):
    def __init__(self, dataset_name: str, dataset_split: str):
        super().__init__()
        self.dataset = load_dataset(path=dataset_name, split=dataset_split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = torch.tensor(np.asarray(self.dataset[idx]['image']))
        label = self.dataset[idx]['label']
        return img, label



