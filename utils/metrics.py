import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def _safe_auc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return np.nan


def _safe_auprc(y_true, y_score):
    try:
        return average_precision_score(y_true, y_score)
    except Exception:
        return np.nan


def evaluate_both(model, dataloader, device, threshold=2.0):
    model.eval()
    y_true_dist, y_true_cls, y_pred_dist = [], [], []
    with torch.no_grad():
        for input_ids, attention_mask, batch_dist, batch_cls in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            preds = model(input_ids, attention_mask).squeeze(-1).cpu().numpy()
            y_pred_dist.extend(preds.tolist())
            y_true_dist.extend(batch_dist.numpy().tolist())
            y_true_cls.extend(batch_cls.numpy().tolist())

    y_true_dist = np.array(y_true_dist, float).ravel()
    y_true_cls = np.array(y_true_cls, int).ravel()
    y_pred_dist = np.array(y_pred_dist, float).ravel()

    mse = mean_squared_error(y_true_dist, y_pred_dist)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_dist, y_pred_dist)
    r2 = r2_score(y_true_dist, y_pred_dist)

    y_pred_bin = (y_pred_dist >= threshold).astype(int)
    acc = accuracy_score(y_true_cls, y_pred_bin)
    f1 = f1_score(y_true_cls, y_pred_bin, zero_division=0)
    prec = precision_score(y_true_cls, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_cls, y_pred_bin, zero_division=0)
    auc = _safe_auc(y_true_cls, y_pred_dist)
    auprc = _safe_auprc(y_true_cls, y_pred_dist)

    reg = {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}
    clf = {"acc": acc, "auc": auc, "auprc": auprc, "f1": f1, "precision": prec, "recall": rec}
    return reg, clf
