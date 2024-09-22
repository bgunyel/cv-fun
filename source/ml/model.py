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
            num_classes=0,
        )

    def forward(self, x):
        h = self.model.forward_features(x)
        return h

