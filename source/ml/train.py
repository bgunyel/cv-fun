from dataclasses import dataclass

from torch.utils.data import DataLoader
from tqdm import tqdm

from source.ml.data import ImageDataset
from source.ml.model import ImageClassifier


@dataclass
class TrainConfig:
    dataset_name: str = 'zh-plus/tiny-imagenet'
    model_name: str = 'timm/inception_v3.tf_adv_in1k'
    n_epochs: int = 2
    batch_size: int = 16


def train_classifier():
    dummy = -32

    train_config = TrainConfig()
    train_data = ImageDataset(dataset_name=train_config.dataset_name, dataset_split='train')
    validation_data = ImageDataset(dataset_name=train_config.dataset_name, dataset_split='valid')

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    model = ImageClassifier(model_name=train_config.model)

    for epoch in range(train_config.n_epochs):
        tqdm_train_loader = tqdm(train_loader, unit="batch", desc=f'Epoch: {epoch}')

        for iteration, (x, y) in enumerate(tqdm_train_loader):

            if iteration % 500 == 0:
                print(f'Iteration: {iteration}')
