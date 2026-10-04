import argparse
import random
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer

from datasets.meld_dataset import MELDManifestDataset
from models.encoders import TextEncoder, AudioEncoder, VisualEncoder
from models.ecerc import ECERCClassifier

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def collate(batch):
    out = {}
    for k in ["input_ids", "attention_mask", "audio", "image", "label"]:
        out[k] = torch.stack([x[k] for x in batch])
    out["dialogue_id"] = [x["dialogue_id"] for x in batch]
    out["utterance_id"] = [x["utterance_id"] for x in batch]
    # For a simple reproducible baseline, current utterances form a one-step
    # context. A dialogue-aware sampler can replace this later.
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    seed_everything(cfg["seed"])

    device = torch.device(
        "cuda" if cfg["device"] == "auto" and torch.cuda.is_available()
        else "cpu"
    )
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(cfg["text_model"])
    ds = MELDManifestDataset(
        args.manifest, tokenizer,
        max_length=cfg["max_text_length"],
        sample_rate=cfg["sample_rate"]
    )

    n_val = max(1, int(0.2 * len(ds)))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg["seed"])
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"],
        shuffle=True, num_workers=cfg["num_workers"],
        collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"],
        shuffle=False, num_workers=cfg["num_workers"],
        collate_fn=collate
    )

    text_encoder = TextEncoder(cfg["text_model"], cfg["hidden_dim"]).to(device)
    audio_encoder = AudioEncoder(cfg["audio_model"], cfg["hidden_dim"]).to(device)
    visual_encoder = VisualEncoder(cfg["hidden_dim"]).to(device)
    model = ECERCClassifier(
        dim=cfg["hidden_dim"],
        num_classes=cfg["num_classes"],
        dropout=cfg["dropout"]
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(text_encoder.parameters()) +
        list(audio_encoder.parameters()) +
        list(visual_encoder.parameters()) +
        list(model.parameters()),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"]
    )
    criterion = torch.nn.CrossEntropyLoss()

    best = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(cfg["epochs"]):
        text_encoder.train(); audio_encoder.train()
        visual_encoder.train(); model.train()
        correct = total = 0

        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            audio = batch["audio"].to(device)
            image = batch["image"].to(device)
            labels = batch["label"].to(device)

            text = text_encoder(ids, mask)
            aud = audio_encoder(audio)
            vis = visual_encoder(image)

            # One-step context baseline. Replace with dialogue window
            # construction for the full conversational experiment.
            context = text.unsqueeze(1)

            out = model(text, aud, vis, context)
            loss = criterion(out["logits"], labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(text_encoder.parameters()) +
                list(audio_encoder.parameters()) +
                list(visual_encoder.parameters()) +
                list(model.parameters()), 1.0
            )
            optimizer.step()

            pred = out["logits"].argmax(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / max(total, 1)

        # Validation
        text_encoder.eval(); audio_encoder.eval()
        visual_encoder.eval(); model.eval()
        vcorrect = vtotal = 0
        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                audio = batch["audio"].to(device)
                image = batch["image"].to(device)
                labels = batch["label"].to(device)

                text = text_encoder(ids, mask)
                aud = audio_encoder(audio)
                vis = visual_encoder(image)
                context = text.unsqueeze(1)
                out = model(text, aud, vis, context)

                pred = out["logits"].argmax(1)
                vcorrect += (pred == labels).sum().item()
                vtotal += labels.size(0)

        val_acc = vcorrect / max(vtotal, 1)
        print(f"Epoch {epoch+1}: train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        if val_acc > best:
            best = val_acc
            torch.save({
                "text_encoder": text_encoder.state_dict(),
                "audio_encoder": audio_encoder.state_dict(),
                "visual_encoder": visual_encoder.state_dict(),
                "model": model.state_dict(),
                "config": cfg,
                "best_val_accuracy": best
            }, "checkpoints/best_model.pt")

if __name__ == "__main__":
    import os
    main()
