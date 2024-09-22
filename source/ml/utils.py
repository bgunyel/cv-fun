import numpy as np
import matplotlib.pyplot as plt

from pydantic import BaseModel

from source.config import settings


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
