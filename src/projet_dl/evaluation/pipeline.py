from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset

from projet_dl.models.simple_ast import SimpleAST
from projet_dl.utils.metrics import compute_icbhi_metrics


def run_evaluation(model_path, data_path, output_dir, batch_size=16, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(data_path, allow_pickle=True)
    specs = torch.from_numpy(data["test_specs"]).float()
    labels = torch.from_numpy(data["test_labels"]).long()
    loader = DataLoader(TensorDataset(specs, labels), batch_size=batch_size, shuffle=False)

    ckpt = torch.load(model_path, map_location=dev)
    model = SimpleAST(num_classes=4).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for batch_specs, batch_labels in loader:
            batch_specs = batch_specs.to(dev).unsqueeze(1)
            logits = model(batch_specs)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            y_true.extend(batch_labels.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    metrics = compute_icbhi_metrics(y_true, y_pred)

    cm = metrics["confusion_matrix"]
    class_names = ["normal", "crackle", "wheeze", "both"]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    cm_path = output_dir / "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()

    metrics_path = output_dir / "metrics.json"
    payload = {
        "sensitivity": metrics["sensitivity"],
        "specificity": metrics["specificity"],
        "icbhi_score": metrics["icbhi_score"],
        "sensitivities_per_class": metrics["sensitivities_per_class"],
        "specificities_per_class": metrics["specificities_per_class"],
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_true, y_pred, target_names=class_names, digits=4),
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    pred_path = output_dir / "predictions.npz"
    np.savez_compressed(pred_path, y_true=np.array(y_true), y_pred=np.array(y_pred), y_probs=np.array(y_probs))

    return {
        "metrics": str(metrics_path),
        "confusion_matrix": str(cm_path),
        "predictions": str(pred_path),
        "icbhi_score": metrics["icbhi_score"],
    }
