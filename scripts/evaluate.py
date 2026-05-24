"""Evaluation script for Plant Disease Identification models."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from src.config.disease_classes import CLASS_NAMES, NUM_CLASSES
from src.dataset import build_dataloaders
from src.models import build_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on test split.")
    parser.add_argument("--model", type=str, choices=["resnet50", "mobilenet_v2"], required=True)
    parser.add_argument("--data-dir", type=str, default="dataset_v2/splits")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_out_dir = Path(args.output_dir) / args.model
    model_out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else (model_out_dir / "best.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    loaders = build_dataloaders(
        splits_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = loaders["test"]

    model = build_model(args.model, num_classes=NUM_CLASSES, pretrained=False).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = nn.CrossEntropyLoss()

    all_labels = []
    all_preds = []
    all_probs = []
    total_loss = 0.0
    total_count = 0
    total_correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_count += bs
            total_correct += (preds == labels).sum().item()

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    test_loss = total_loss / max(total_count, 1)
    test_acc = total_correct / max(total_count, 1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    report_text = classification_report(
        all_labels, all_preds, target_names=CLASS_NAMES, digits=4, zero_division=0
    )
    report_dict = classification_report(
        all_labels, all_preds, target_names=CLASS_NAMES, digits=4, zero_division=0, output_dict=True
    )

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))

    # Save classification report txt
    with open(model_out_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Checkpoint: {ckpt_path}\n")
        f.write(f"Test loss: {test_loss:.6f}\n")
        f.write(f"Test accuracy: {test_acc:.6f}\n")
        f.write(f"Precision (weighted): {precision:.6f}\n")
        f.write(f"Recall (weighted): {recall:.6f}\n")
        f.write(f"F1-score (weighted): {f1:.6f}\n\n")
        f.write(report_text)

    # Save classification report csv
    with open(model_out_dir / "classification_report.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1-score", "support"])
        for k, v in report_dict.items():
            if isinstance(v, dict):
                writer.writerow([k, v.get("precision", ""), v.get("recall", ""), v.get("f1-score", ""), v.get("support", "")])

    # Save predictions csv
    with open(model_out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["y_true_idx", "y_true_name", "y_pred_idx", "y_pred_name"] + [f"prob_{c}" for c in CLASS_NAMES]
        writer.writerow(header)
        for yt, yp, prob in zip(all_labels, all_preds, all_probs):
            writer.writerow([yt, CLASS_NAMES[yt], yp, CLASS_NAMES[yp]] + prob)

    # Plot confusion matrix
    fig_w = max(10, NUM_CLASSES * 0.8)
    fig_h = max(8, NUM_CLASSES * 0.7)
    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {args.model}")
    plt.colorbar()

    tick_marks = np.arange(NUM_CLASSES)
    plt.xticks(tick_marks, CLASS_NAMES, rotation=90)
    plt.yticks(tick_marks, CLASS_NAMES)

    thresh = cm.max() / 2.0 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=7,
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(model_out_dir / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Test loss: {test_loss:.6f}")
    print(f"Test accuracy: {test_acc:.6f}")
    print(f"Precision (weighted): {precision:.6f}")
    print(f"Recall (weighted): {recall:.6f}")
    print(f"F1-score (weighted): {f1:.6f}")
    print(f"Saved outputs to: {model_out_dir}")


if __name__ == "__main__":
    main()