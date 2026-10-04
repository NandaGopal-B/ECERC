"""
ECERC-inspired multimodal emotion recognition architecture.

The implementation is intentionally modular so each research component can
be enabled/disabled for ablation studies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidenceGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.gate(x)


class CauseEncoder(nn.Module):
    """Encodes the conversational context preceding the current utterance."""
    def __init__(self, dim, heads=4, layers=1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, batch_first=True,
            dropout=0.1, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)

    def forward(self, context):
        # context: [B, T, D]
        return self.encoder(context)


class EvidenceCauseAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, evidence, cause):
        # Query = current evidence; Key/Value = contextual causes.
        query = evidence.unsqueeze(1)
        out, weights = self.attn(query, cause, cause, need_weights=True)
        out = self.norm(query + out).squeeze(1)
        return out, weights


class FeatureGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.proj(x)


class ECERCClassifier(nn.Module):
    """
    Inputs:
        text_features:  [B, D]
        audio_features: [B, D]
        visual_features:[B, D]
        context_features:[B, T, D]
    """
    def __init__(self, dim=256, num_classes=7, dropout=0.2,
                 use_evidence=True, use_cause=True,
                 use_attention=True, use_feature_gate=True):
        super().__init__()
        self.dim = dim
        self.use_evidence = use_evidence
        self.use_cause = use_cause
        self.use_attention = use_attention
        self.use_feature_gate = use_feature_gate

        self.fusion = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.evidence_gate = EvidenceGate(dim)
        self.cause_encoder = CauseEncoder(dim)
        self.ec_attention = EvidenceCauseAttention(dim)
        self.feature_gate = FeatureGate(dim)

        self.classifier = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes)
        )

    def forward(self, text, audio, visual, context):
        fused = self.fusion(torch.cat([text, audio, visual], dim=-1))

        evidence = self.evidence_gate(fused) if self.use_evidence else fused

        if self.use_cause:
            cause = self.cause_encoder(context)
        else:
            cause = context

        if self.use_attention:
            interaction, attention = self.ec_attention(evidence, cause)
        else:
            interaction = cause.mean(dim=1)
            attention = None

        candidate = torch.cat([evidence, interaction], dim=-1)

        if self.use_feature_gate:
            candidate = torch.cat([
                self.feature_gate(evidence),
                self.feature_gate(interaction)
            ], dim=-1)

        logits = self.classifier(candidate)
        return {
            "logits": logits,
            "evidence": evidence,
            "interaction": interaction,
            "attention": attention
        }


class ModalityEncoder(nn.Module):
    """Lightweight projection for precomputed modality features."""
    def __init__(self, input_dim, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )

    def forward(self, x):
        return self.net(x)
