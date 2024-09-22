from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from timm.data import resolve_model_data_config, create_transform
from tqdm import tqdm

from source.ml.data import ImageDataset, generate_train_validation_datasets
from source.ml.model import ImageClassifier


@dataclass
class TrainConfig:
    dataset_name: str = 'stanford_cars'
    model_name: str = 'timm/inception_v3.tf_adv_in1k'
    n_epochs: int = 2
    batch_size: int = 16


def train_classifier(device: torch.device):
    dummy = -32

    train_config = TrainConfig()

    ds_train, ds_valid = generate_train_validation_datasets(dataset_name=train_config.dataset_name,
                                                            validation_set_ratio=0.2)

    t = ds_train.__getitem__(0)

    train_loader = DataLoader(
        dataset=ds_train,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    model = ImageClassifier(model_name=train_config.model_name).to(device)
    data_config = resolve_model_data_config(model=model, args={'input_size': (3, 299, 299)})
    transforms = create_transform(**data_config, is_training=True)

    for epoch in range(train_config.n_epochs):
        tqdm_train_loader = tqdm(train_loader, unit="batch", desc=f'Epoch: {epoch}')

        for iteration, data in enumerate(tqdm_train_loader):
            x = data['image'].to(device)
            label = data['label'].to(device)

            x = transforms(x)
            logits = model(x)

            dummy = -32




    dummy = -32
