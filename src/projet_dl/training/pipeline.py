from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from projet_dl.models.simple_ast import SimpleAST
from projet_dl.training.losses import FocalLoss, class_weights_from_labels
from projet_dl.training.sam import SAM
from projet_dl.utils.metrics import compute_icbhi_metrics


def _train_epoch(model, loader, criterion, optimizer, device, use_sam):
    model.train()
    total_loss = 0.0
    preds_all, labels_all = [], []

    for specs, labels in tqdm(loader, desc="train", leave=False):
        specs = specs.to(device).unsqueeze(1)
        labels = labels.to(device)

        if use_sam:
            def closure():
                optimizer.zero_grad()
                logits = model(specs)
                loss = criterion(logits, labels)
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            with torch.no_grad():
                logits = model(specs)
        else:
            optimizer.zero_grad()
            logits = model(specs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        preds = logits.argmax(dim=1)
        preds_all.extend(preds.detach().cpu().numpy())
        labels_all.extend(labels.detach().cpu().numpy())

    metrics = compute_icbhi_metrics(labels_all, preds_all)
    return total_loss / max(1, len(loader)), metrics


def _eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds_all, labels_all = [], []

    with torch.no_grad():
        for specs, labels in tqdm(loader, desc="val", leave=False):
            specs = specs.to(device).unsqueeze(1)
            labels = labels.to(device)
            logits = model(specs)
            loss = criterion(logits, labels)

            total_loss += float(loss.item())
            preds = logits.argmax(dim=1)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    metrics = compute_icbhi_metrics(labels_all, preds_all)
    return total_loss / max(1, len(loader)), metrics


def run_training(data_path, output_path, epochs=20, batch_size=8, lr=1e-5, use_sam=True, use_focal=True, seed=42, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)

    data = np.load(data_path, allow_pickle=True)
    train_specs = torch.from_numpy(data["train_specs"]).float()
    train_labels = torch.from_numpy(data["train_labels"]).long()
    test_specs = torch.from_numpy(data["test_specs"]).float()
    test_labels = torch.from_numpy(data["test_labels"]).long()

    train_loader = DataLoader(TensorDataset(train_specs, train_labels), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(test_specs, test_labels), batch_size=batch_size, shuffle=False)

    model = SimpleAST(num_classes=4).to(dev)
    weights = class_weights_from_labels(train_labels, num_classes=4).to(dev)
    criterion = FocalLoss(alpha=weights, gamma=2.0) if use_focal else nn.CrossEntropyLoss(weight=weights)

    if use_sam:
        optimizer = SAM(model.parameters(), base_optimizer=AdamW, lr=lr, weight_decay=1e-4, rho=0.05)
        scheduler = CosineAnnealingLR(optimizer.base_optimizer, T_max=epochs)
    else:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_score = -1.0
    history = {"train_loss": [], "val_loss": [], "val_icbhi": []}

    for epoch in range(epochs):
        train_loss, _ = _train_epoch(model, train_loader, criterion, optimizer, dev, use_sam)
        val_loss, val_metrics = _eval_epoch(model, val_loader, criterion, dev)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_icbhi"].append(val_metrics["icbhi_score"])

        print(
            f"[Epoch {epoch+1}/{epochs}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"Se={val_metrics['sensitivity']:.4f} Sp={val_metrics['specificity']:.4f} "
            f"ICBHI={val_metrics['icbhi_score']:.4f}"
        )

        if val_metrics["icbhi_score"] > best_score:
            best_score = val_metrics["icbhi_score"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict() if use_sam else optimizer.state_dict(),
                    "best_icbhi": best_score,
                    "metrics": val_metrics,
                },
                output_path,
            )

    history_path = output_path.parent / "training_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return {
        "best_icbhi": best_score,
        "checkpoint": str(output_path),
        "history": str(history_path),
    }
