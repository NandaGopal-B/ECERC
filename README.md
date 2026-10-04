# ECERC: Evidence-Cause Attention Network for Multi-Modal Emotion Recognition in Conversation

A complete academic implementation of an ECERC-inspired multimodal emotion recognition system for conversational data.

The project combines:
- Text representations using a Transformer encoder
- Audio representations using a Wav2Vec2 encoder
- Visual representations using a ResNet encoder
- Evidence gating
- Conversation/cause encoding
- Evidence-cause cross attention
- Feature gating
- Seven-class emotion classification
- Baseline and ablation experiments
- Interactive Streamlit demo

## Emotion classes

`anger, disgust, sadness, joy, neutral, surprise, fear`

## Repository structure

```text
ECERC-Emotion-Recognition/
├── app/                    # Streamlit application
├── datasets/               # Dataset loading utilities
├── evaluation/             # Metrics and evaluation
├── experiments/            # Baselines and ablations
├── models/                 # ECERC architecture
├── preprocessing/          # Feature preparation
├── training/               # Training and validation
├── notebooks/              # Notebook-ready workflow
├── reports/                # Report and figures
├── data/                   # Put downloaded datasets here
├── checkpoints/            # Saved model checkpoints
├── requirements.txt
├── config.yaml
└── README.md
```

## Dataset

This repository does **not** redistribute MELD/IEMOCAP media files. Download the dataset from its official source and place it under `data/raw/`.

For the first run, the project supports a CSV manifest with:

```text
dialogue_id,utterance_id,speaker,text,audio_path,video_path,label
```

If audio/video paths are unavailable, the text-only fallback can still be used for development.

## Installation

Recommended Python: 3.10 or 3.11.

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
```

## Quick demo

The Streamlit application can run in demo mode without a dataset:

```bash
streamlit run app/streamlit_app.py
```

## Training

Prepare a manifest:

```text
data/processed/meld_manifest.csv
```

Then:

```bash
python training/train.py --config config.yaml --manifest data/processed/meld_manifest.csv
```

Evaluate:

```bash
python evaluation/evaluate.py --checkpoint checkpoints/best_model.pt --manifest data/processed/meld_manifest.csv
```

## Important academic note

This repository is an implementation inspired by the ECERC research architecture. The original research work should be cited in academic submissions. Do not claim the original architecture as independently invented.

For a final submission, report the actual results obtained from your own training runs rather than inserting fabricated accuracy/F1 values.

## Suggested experiments

1. Text-only baseline
2. Audio-only baseline
3. Visual-only baseline
4. Text + Audio
5. Text + Audio + Visual
6. ECERC full model
7. ECERC without evidence gating
8. ECERC without cause encoding
9. ECERC without evidence-cause attention
10. ECERC without feature gating

## Citation

Zhang, Tao and Tan, Zhenhua. ECERC: Evidence-Cause Attention Network for Multi-Modal Emotion Recognition in Conversation. ACL 2025.

Official paper:
https://aclanthology.org/2025.acl-long.102/

Official implementation:
https://github.com/TAN-OpenLab/ECERC
