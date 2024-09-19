import torch
import torch.nn as nn
import timm


# TODO: INCOMPLETE
class ImageClassifier(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()

        self.model = timm.create_model(
            model_name=model_name,
            pretrained=True,
            features_only=True,
        )

    def forward(self, x):
        h = self.model(x)
        return h

