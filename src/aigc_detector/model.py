from __future__ import annotations

import torch
from torch import nn
from transformers import CLIPVisionModel


class ClipBinaryDetector(nn.Module):
    """Frozen CLIP vision encoder with a small, trainable binary head."""

    def __init__(self, model_name: str, freeze_encoder: bool = True) -> None:
        super().__init__()
        self.encoder = CLIPVisionModel.from_pretrained(model_name)
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(pixel_values=pixel_values)
        return self.classifier(encoded.pooler_output).squeeze(-1)
