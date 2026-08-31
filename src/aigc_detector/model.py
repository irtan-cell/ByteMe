"""Frozen CLIP vision encoder with a small trainable MLP head."""
import torch
import torch.nn as nn
from transformers import CLIPVisionModel

BACKBONE = "openai/clip-vit-base-patch16"


class Detector(nn.Module):
    def __init__(self, backbone=BACKBONE, hidden=512, dropout=0.2, unfreeze_last=0):
        super().__init__()
        self.encoder = CLIPVisionModel.from_pretrained(backbone)
        for p in self.encoder.parameters():
            p.requires_grad = False
        if unfreeze_last:
            for layer in self.encoder.vision_model.encoder.layers[-unfreeze_last:]:
                for p in layer.parameters():
                    p.requires_grad = True
        dim = self.encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def embed(self, pixel_values):
        out = self.encoder(pixel_values=pixel_values)
        return out.pooler_output

    def forward(self, pixel_values):
        return self.head(self.embed(pixel_values)).squeeze(-1)
