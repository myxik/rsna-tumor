import timm
import torch
import torch.nn as nn

from torch import Tensor


class GRUover2d(nn.Module):
    def __init__(
        self, model_name: str, pretrained: bool, input_size: int, hidden_size: int
    ) -> None:
        super().__init__()
        self.conv_model = timm.create_model(model_name, pretrained=pretrained)
        if "efficientnet" in model_name:  # == "tf_efficientnetv2_xl_in21k"
            self.conv_model.classifier = nn.Linear(
                self.conv_model.classifier.in_features, input_size
            )
        self.hidden_size = hidden_size
        self.gru_layer = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True,
            bias=False,
        )

        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_size, 2048),
            nn.Linear(2048, 1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 1),
        )

    def init_hidden(self, batch_size: int, type: str) -> Tensor:
        h_state = torch.empty(2, batch_size, self.hidden_size).type(type)
        nn.init.uniform_(h_state, -0.5, 0.5)
        return h_state

    def forward(self, x: Tensor) -> Tensor:
        h_state = self.init_hidden(1, x.type())

        feature_set = self.conv_model(x)[
            None, ...
        ]  # (1, seq_len, num_features) seq_len=sequential images
        for _ in range(feature_set.shape[1]):
            output, h_state = self.gru_layer(feature_set, h_state)

        output = output.squeeze(0)
        output = output.view(output.shape[0], -1)
        out = self.classifier(output)
        return out
