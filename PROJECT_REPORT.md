# ECERC Project Report

## 1. Title

**ECERC: Evidence-Cause Attention Network for Multi-Modal Emotion Recognition in Conversation**

## 2. Abstract

Emotion recognition in conversation is challenging because emotional meaning is influenced not only by the words spoken by a participant but also by vocal characteristics, facial expressions, speaker context, and preceding utterances. This project implements an ECERC-inspired multimodal emotion recognition system that integrates text, audio and visual information. The architecture uses evidence gating to emphasize informative multimodal features, contextual cause encoding to represent conversational history, evidence-cause attention to model relationships between emotional evidence and contextual causes, and feature gating before classification. The system predicts seven conversational emotions: anger, disgust, sadness, joy, neutral, surprise and fear. An interactive Streamlit dashboard is included for demonstration and explainability.

## 3. Problem Statement

Traditional text-only emotion classifiers may fail when the emotional meaning of an utterance depends on tone, facial expression or preceding conversation. A multimodal conversational model is therefore required to jointly analyze linguistic, acoustic and visual cues while considering contextual causes.

## 4. Objectives

1. Build a multimodal conversational emotion recognition pipeline.
2. Extract text, audio and visual representations.
3. Implement evidence gating.
4. Encode conversational context.
5. Implement evidence-cause attention.
6. Compare multimodal and unimodal baselines.
7. Conduct ablation experiments.
8. Build an interactive Streamlit demonstration.

## 5. Dataset

The recommended benchmark is MELD. The dataset contains multimodal conversational utterances with emotion annotations. The media files are not redistributed with this repository and must be obtained from the official dataset source.

## 6. Methodology

### Text
A Transformer encoder generates a semantic representation of each utterance.

### Audio
Wav2Vec2 extracts speech representations.

### Visual
A ResNet encoder extracts visual features from representative frames.

### Evidence Gating
A learnable gate emphasizes features that are useful as emotional evidence.

### Cause Encoding
Previous conversational information is encoded as contextual causes.

### Evidence-Cause Attention
Cross attention relates the current emotional evidence to contextual causes.

### Feature Gating
The resulting candidate representation is filtered before classification.

### Classification
A multilayer perceptron predicts one of seven emotion categories.

## 7. Experiments

Report actual results from your runs.

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Text-only | TO BE MEASURED | TO BE MEASURED | TO BE MEASURED |
| Audio-only | TO BE MEASURED | TO BE MEASURED | TO BE MEASURED |
| Visual-only | TO BE MEASURED | TO BE MEASURED | TO BE MEASURED |
| Text + Audio | TO BE MEASURED | TO BE MEASURED | TO BE MEASURED |
| Text + Audio + Visual | TO BE MEASURED | TO BE MEASURED | TO BE MEASURED |
| ECERC | TO BE MEASURED | TO BE MEASURED | TO BE MEASURED |

## 8. Ablation Study

| Variant | Macro F1 |
|---|---:|
| Full ECERC | TO BE MEASURED |
| Without evidence gating | TO BE MEASURED |
| Without cause encoding | TO BE MEASURED |
| Without evidence-cause attention | TO BE MEASURED |
| Without feature gating | TO BE MEASURED |

## 9. Limitations

The full multimodal system requires substantial compute and storage because pretrained audio and visual encoders are large. Video preprocessing is also more expensive than text processing. Missing or noisy modalities can affect performance.

## 10. Future Work

- Better dialogue-window construction
- Speaker-aware context encoding
- Real-time audio/video inference
- Larger multimodal foundation models
- More advanced causal discovery
- Human-centered explainability
- Cross-dataset generalization

## 11. Conclusion

This project demonstrates an end-to-end research-oriented approach to multimodal emotion recognition in conversation. By combining linguistic, acoustic and visual evidence with conversational context and attention mechanisms, the system provides a stronger framework than a conventional single-modality classifier.
