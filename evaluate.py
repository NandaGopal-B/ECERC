import argparse
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import confusion_matrix
import numpy as np

from datasets.meld_dataset import MELDManifestDataset
from models.encoders import TextEncoder, AudioEncoder, VisualEncoder
from models.ecerc import ECERCClassifier
from evaluation.metrics import calculate_metrics

def collate(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "audio": torch.stack([x["audio"] for x in batch]),
        "image": torch.stack([x["image"] for x in batch]),
        "label": torch.stack([x["label"] for x in batch])
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device(
        "cuda" if cfg["device"] == "auto" and torch.cuda.is_available()
        else "cpu"
    )
    ckpt = torch.load(args.checkpoint, map_location=device)

    tokenizer = AutoTokenizer.from_pretrained(cfg["text_model"])
    ds = MELDManifestDataset(args.manifest, tokenizer,
                             cfg["max_text_length"], cfg["sample_rate"])
    loader = DataLoader(ds, batch_size=cfg["batch_size"],
                        shuffle=False, collate_fn=collate)

    te = TextEncoder(cfg["text_model"], cfg["hidden_dim"]).to(device)
    ae = AudioEncoder(cfg["audio_model"], cfg["hidden_dim"]).to(device)
    ve = VisualEncoder(cfg["hidden_dim"]).to(device)
    model = ECERCClassifier(
        cfg["hidden_dim"], cfg["num_classes"], cfg["dropout"]
    ).to(device)

    te.load_state_dict(ckpt["text_encoder"])
    ae.load_state_dict(ckpt["audio_encoder"])
    ve.load_state_dict(ckpt["visual_encoder"])
    model.load_state_dict(ckpt["model"])

    te.eval(); ae.eval(); ve.eval(); model.eval()
    ys, ps = [], []

    with torch.no_grad():
        for b in loader:
            ids = b["input_ids"].to(device)
            mask = b["attention_mask"].to(device)
            audio = b["audio"].to(device)
            image = b["image"].to(device)

            text = te(ids, mask)
            aud = ae(audio)
            vis = ve(image)
            out = model(text, aud, vis, text.unsqueeze(1))

            ys.extend(b["label"].numpy().tolist())
            ps.extend(out["logits"].argmax(1).cpu().numpy().tolist())

    metrics = calculate_metrics(ys, ps)
    print(metrics)
    print("Confusion matrix:")
    print(confusion_matrix(ys, ps))

if __name__ == "__main__":
    main()
