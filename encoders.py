"""
Transformer, audio and visual feature encoders.

These encoders use Hugging Face/torchvision models and project their
representations to a common ECERC dimension.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, Wav2Vec2Model
from torchvision.models import resnet18, ResNet18_Weights


class TextEncoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", output_dim=256):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hidden, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )

    def forward(self, input_ids, attention_mask):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state[:, 0]
        return self.proj(out)


class AudioEncoder(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base", output_dim=256):
        super().__init__()
        self.backbone = Wav2Vec2Model.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hidden, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )

    def forward(self, input_values, attention_mask=None):
        out = self.backbone(
            input_values=input_values,
            attention_mask=attention_mask
        ).last_hidden_state.mean(dim=1)
        return self.proj(out)


class VisualEncoder(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.DEFAULT)
        hidden = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.proj = nn.Sequential(
            nn.Linear(hidden, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )

    def forward(self, images):
        # images: [B, 3, H, W]
        return self.proj(self.backbone(images))
