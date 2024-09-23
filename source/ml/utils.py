import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pydantic import BaseModel

from source.config import settings, model_settings


def show_image(image):
    plt.figure()
    plt.imshow(np.asarray(image))
    plt.show()


def get_dataset_config(dataset_name: str) -> BaseModel:

    dataset_config = {
        'stanford-cars': settings.STANFORD_CARS,
        'stanford_cars': settings.STANFORD_CARS,
        'tiny-imagenet': settings.TINY_IMAGENET,
    }

    return dataset_config[dataset_name]


def get_model_config(model_name: str) -> BaseModel:

    model_configs = {
        'inception_v3': model_settings.INCEPTION_V3,
        'inception-v3': model_settings.INCEPTION_V3,
    }

    return model_configs[model_name]


@torch.no_grad()
def evaluate(model: nn.Module, data_loader: DataLoader):

    device = torch.device(model_settings.DEVICE)
    model.eval()

    losses = torch.zeros(len(data_loader))

    for idx, data in enumerate(data_loader):
        x = data['image'].to(device)
        label = data['label'].to(device)

        with torch.autocast(device_type=model_settings.DEVICE):
            logits = model(x)
            loss = F.cross_entropy(input=logits, target=label)

        losses[idx] = loss.item()

    model.train()
    return losses.mean()

