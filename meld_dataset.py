import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio
from PIL import Image
from torchvision import transforms

EMOTIONS = ["anger", "disgust", "sadness", "joy", "neutral", "surprise", "fear"]
LABEL2ID = {x: i for i, x in enumerate(EMOTIONS)}

class MELDManifestDataset(Dataset):
    """
    CSV columns:
    dialogue_id, utterance_id, speaker, text, audio_path, video_path, label

    Audio/video are optional during development. Missing modalities are replaced
    by zero tensors so the pipeline remains testable.
    """
    def __init__(self, csv_path, tokenizer, max_length=128, sample_rate=16000):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sample_rate = sample_rate
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.df)

    def _audio(self, path):
        if not isinstance(path, str) or not os.path.exists(path):
            return torch.zeros(self.sample_rate)
        waveform, sr = torchaudio.load(path)
        waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sr, self.sample_rate
            )
        return waveform.squeeze(0)

    def _image(self, path):
        if not isinstance(path, str) or not os.path.exists(path):
            return torch.zeros(3, 224, 224)
        return self.transform(Image.open(path).convert("RGB"))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["text"])
        tok = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        label = row["label"]
        if isinstance(label, str):
            label = LABEL2ID[label.lower()]

        return {
            "input_ids": tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0),
            "audio": self._audio(row.get("audio_path", "")),
            "image": self._image(row.get("video_path", "")),
            "label": torch.tensor(int(label), dtype=torch.long),
            "dialogue_id": str(row["dialogue_id"]),
            "utterance_id": str(row["utterance_id"])
        }
