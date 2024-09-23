from io import BytesIO

import numpy as np
import polars as pl
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from source.ml.utils import get_dataset_config


class ImageDataset(Dataset):
    def __init__(self, image_list: list[dict], output_width: int=None, output_height: int=None):
        super().__init__()
        self.image_list = image_list
        self.output_width = output_width
        self.output_height = output_height

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        label = self.image_list[idx]['label']
        img_bytes = BytesIO(self.image_list[idx]['image']['bytes'])
        img = Image.open(img_bytes).convert("RGB")

        if (self.output_width is not None) and (self.output_height is not None):
            img = img.resize((self.output_width, self.output_height))

        width = img.width
        height = img.height
        n_channels = 3  # we convert to RGB image mode

        img_tensor = torch.tensor(np.asarray(img), dtype=torch.float).view(n_channels, height, width)

        return {'image': img_tensor, 'label': label}



def generate_train_validation_datasets(
        dataset_name: str,
        image_width: int,
        image_height: int,
        validation_set_ratio: float = 0.2
) -> tuple[ImageDataset, ImageDataset]:

    dataset_config = get_dataset_config(dataset_name=dataset_name)
    df = pl.read_parquet(dataset_config.TRAIN_FILE)
    rows = df.rows(named=True)

    idx_train, idx_valid = train_test_split(
        range(len(rows)),
        test_size=validation_set_ratio,
        shuffle=True,
        random_state=1881,
        stratify=df.get_column('label').to_list()
    )
    rows_train = [rows[i] for i in idx_train]
    rows_valid = [rows[i] for i in idx_valid]

    ds_train = ImageDataset(image_list=rows_train,
                            output_width=image_width,
                            output_height=image_height)
    ds_valid = ImageDataset(image_list=rows_valid,
                            output_width=image_width,
                            output_height=image_height)

    return ds_train, ds_valid
