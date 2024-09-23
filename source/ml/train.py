from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from timm.data import resolve_model_data_config, create_transform
from tqdm import tqdm

from source.ml.data import ImageDataset, generate_train_validation_datasets
from source.ml.model import ImageClassifier
from source.ml.utils import get_model_config, get_dataset_config, evaluate
from source.config import model_settings


@dataclass
class OptimizerConfig:
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8


@dataclass
class TrainConfig:
    dataset_name: str = 'stanford_cars'
    model_name: str = 'inception-v3'
    n_epochs: int = 10
    batch_size: int = 32


def train_classifier():

    device = torch.device(model_settings.DEVICE)
    torch.manual_seed(seed=1919)

    train_config = TrainConfig()
    optimizer_config = OptimizerConfig()
    dataset_config = get_dataset_config(dataset_name=train_config.dataset_name)
    model_config = get_model_config(model_name=train_config.model_name)

    ds_train, ds_valid = generate_train_validation_datasets(dataset_name=train_config.dataset_name,
                                                            image_width=model_config.INPUT_DIMS[1],
                                                            image_height=model_config.INPUT_DIMS[2],
                                                            validation_set_ratio=0.2)

    train_loader = DataLoader(
        dataset=ds_train,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )



    model = ImageClassifier(model_config=model_config, num_classes=dataset_config.NUM_CLASSES).to(device)
    # data_config = resolve_model_data_config(model=model, args={'input_size': (3, 299, 299)})
    # transforms = create_transform(**data_config, is_training=True)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=optimizer_config.lr,
                                  weight_decay=optimizer_config.weight_decay,
                                  betas=optimizer_config.betas,
                                  eps=optimizer_config.eps)

    model.train()
    epoch_iters = len(train_loader)
    eval_iters = int(epoch_iters * 0.1)

    for epoch in range(train_config.n_epochs):
        tqdm_train_loader = tqdm(train_loader, unit="batch", desc=f'Epoch: {epoch}')

        for iteration, data in enumerate(tqdm_train_loader):
            x = data['image'].to(device)
            label = data['label'].to(device)
            optimizer.zero_grad()

            with torch.autocast(device_type=model_settings.DEVICE):
                logits = model(x)
                loss = F.cross_entropy(input=logits, target=label)

            loss.backward()
            optimizer.step()

            if (iteration % eval_iters == 0) or (iteration == epoch_iters - 1):

                train_set_loss = evaluate(
                    model=model,
                    data_loader=DataLoader(dataset=ds_train,
                                            batch_size=train_config.batch_size,
                                            shuffle=False,
                                            num_workers=4,
                                            pin_memory=True,
                                            drop_last=False)
                )

                valid_set_loss = evaluate(
                    model=model,
                    data_loader=DataLoader(dataset=ds_valid,
                                           batch_size=train_config.batch_size,
                                           shuffle=False,
                                           num_workers=4,
                                           pin_memory=True,
                                           drop_last=False)
                )

                print(f"\tEpoch {epoch}\t Iteration {iteration}\t Batch Loss {loss.item():.4f} | Train Set Loss {train_set_loss:.4f} | Valid Set Loss {valid_set_loss:.4f} ")

            dummy = -32




    dummy = -32
