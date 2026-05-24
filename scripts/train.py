"""Training script for Plant Disease Identification (ResNet50/MobileNetV2)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.config.disease_classes import NUM_CLASSES, CLASS_NAMES
from src.dataset import build_dataloaders
from src.models import build_model


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


def run_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    training: bool = True,
) -> Tuple[float, float]:
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
        total_count += batch_size

    epoch_loss = total_loss / max(total_count, 1)
    epoch_acc = total_correct / max(total_count, 1)
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description="Train a plant disease classifier.")
    parser.add_argument("--model", type=str, choices=["resnet50", "mobilenet_v2"], required=True)
    parser.add_argument("--data-dir", type=str, default="dataset_v2/splits")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--optimizer", type=str, choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--scheduler", type=str, choices=["plateau", "cosine"], default="plateau")
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--freeze-epochs", type=int, default=3)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model} | NUM_CLASSES: {NUM_CLASSES}")

    loaders = build_dataloaders(
        splits_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader, val_loader = loaders["train"], loaders["val"]

    model = build_model(args.model, num_classes=NUM_CLASSES, pretrained=True).to(device)

    # Freeze backbone for warmup head training
    if args.freeze_epochs > 0:
        if args.model == "resnet50":
            for p in model.parameters():
                p.requires_grad = False
            for p in model.fc.parameters():
                p.requires_grad = True
        else:
            for p in model.parameters():
                p.requires_grad = False
            for p in model.classifier.parameters():
                p.requires_grad = True

    criterion = nn.CrossEntropyLoss()

    def make_optimizer(lr: float):
        params = [p for p in model.parameters() if p.requires_grad]
        if args.optimizer == "adamw":
            return AdamW(params, lr=lr, weight_decay=1e-4)
        return SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)

    optimizer = make_optimizer(args.lr)

    def make_scheduler(opt):
        if args.scheduler == "plateau":
            return ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
        return CosineAnnealingLR(opt, T_max=max(args.epochs, 1))

    scheduler = make_scheduler(optimizer)

    model_out_dir = Path(args.output_dir) / args.model
    model_out_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_val_acc = -1.0
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch [{epoch}/{args.epochs}]")

        # Unfreeze after freeze stage
        if epoch == args.freeze_epochs + 1 and args.freeze_epochs > 0:
            print("Unfreezing full model for fine-tuning...")
            for p in model.parameters():
                p.requires_grad = True
            optimizer = make_optimizer(args.finetune_lr)
            scheduler = make_scheduler(optimizer)

        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, optimizer, device, training=True)
        val_loss, val_acc = run_one_epoch(model, val_loader, criterion, optimizer, device, training=False)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_acc)
        else:
            scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "lr": lr_now,
        }
        history.append(row)

        print(
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | lr={lr_now:.6g}"
        )

        # Save last checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_name": args.model,
                "num_classes": NUM_CLASSES,
                "class_names": CLASS_NAMES,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            },
            model_out_dir / "last.pt",
        )

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_name": args.model,
                    "num_classes": NUM_CLASSES,
                    "class_names": CLASS_NAMES,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                },
                model_out_dir / "best.pt",
            )
            print(f"Saved best checkpoint with val_acc={val_acc:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

        # Early stopping
        if epochs_no_improve >= args.patience:
            print(f"Early stopping triggered (patience={args.patience}).")
            break

    # Save history
    history_csv = model_out_dir / "history.csv"
    with open(history_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "lr"])
        writer.writeheader()
        writer.writerows(history)

    with open(model_out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val_acc={best_val_acc:.4f}")
    print(f"Artifacts saved to: {model_out_dir}")


if __name__ == "__main__":
    main()