import torch
import torch.nn as nn
import timm

from pydantic import BaseModel


class ImageClassifier(nn.Module):
    def __init__(self, model_config: BaseModel, num_classes: int):
        super().__init__()
        self.feature_map_dims = model_config.FEATURE_MAP_DIMS

        self.model = timm.create_model(
            model_name=model_config.MODEL_NAME,
            pretrained=True,
            num_classes=0, # To get the feature map
        )
        self.average_pool = nn.AvgPool2d(kernel_size=(self.feature_map_dims[1], self.feature_map_dims[2]))
        self.fc_last = nn.Linear(in_features=self.feature_map_dims[0], out_features=model_config.HIDDEN_LAYER_SIZE)
        self.fc_last_dropout = nn.Dropout(p=model_config.DROP_OUT_PROB)
        self.fc_logits = nn.Linear(in_features=model_config.HIDDEN_LAYER_SIZE, out_features=num_classes)

    def forward(self, x):
        h = self.model.forward_features(x)
        out = self.average_pool(h)
        out = torch.squeeze(out)
        out = self.fc_last(out)
        out = self.fc_last_dropout(out)
        out = self.fc_logits(out)
        return out  # logits

