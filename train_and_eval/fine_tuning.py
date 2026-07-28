import os

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForSequenceClassification

from configs.config import BacktestConfig, CrossValidationConfig
from models.model import CrossEncoder
from train_and_eval.training_strategies import (
    make_cross_validation_splits,
    split_fludataset_for_backtest,
)
from utils.data_processing import FLUDataset, H3N2DataProcessor
from utils.metrics import evaluate_both
from utils.recorders import BacktestRecorder, CVRecoder
from utils.utils import ensure_dir, get_device, set_random_seed


def _build_processor_and_datasets(cfg, tokenizer, seq_len):
    processor = H3N2DataProcessor(
        ha_path=cfg.ha_path,
        hi_path=cfg.hi_path,
        extra_ha_path=getattr(cfg, "extra_ha_path", None),
        extra_hi_path=getattr(cfg, "extra_hi_path", None),
        metadata_path=getattr(cfg, "metadata_path", None),
        bin_threshold=cfg.classification_threshold,
        deduplicate=cfg.deduplicate,
    )
    ds_eval = FLUDataset(
        processor, tokenizer=tokenizer, seq_len=seq_len, mode="test"
    )
    ds_train = FLUDataset(
        processor, tokenizer=tokenizer, seq_len=seq_len, mode="train"
    )
    return processor, ds_train, ds_eval


def _new_model(pretrained_model_dir, device):
    base = EsmForSequenceClassification.from_pretrained(
        pretrained_model_dir, num_labels=1
    ).to(device)
    return CrossEncoder(base).to(device)


def _train_one_split(
    model, train_loader, val_loader, cfg, device, best_path, tag
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = nn.MSELoss()
    best_val_rmse = float("inf")
    no_improve = 0
    best_epoch = 0
    hist = {"mse": [], "rmse": [], "mae": [], "r2": []}

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        with tqdm(
            train_loader,
            desc=f"{tag} Epoch {epoch}/{cfg.num_epochs}",
            unit="batch",
        ) as progress:
            for input_ids, attention_mask, y_dist, _y_cls in progress:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                y_dist = y_dist.to(device).float().unsqueeze(-1)
                preds = model(input_ids, attention_mask)
                loss = criterion(preds, y_dist)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                progress.set_postfix(loss=f"{loss.item():.4f}")

        val_reg, val_clf = evaluate_both(
            model,
            val_loader,
            device,
            threshold=cfg.classification_threshold,
        )
        scheduler.step(val_reg["rmse"])
        hist["mse"].append(val_reg["mse"])
        hist["rmse"].append(val_reg["rmse"])
        hist["mae"].append(val_reg["mae"])
        hist["r2"].append(val_reg["r2"])

        if val_reg["rmse"] + 1e-9 < best_val_rmse:
            best_val_rmse = val_reg["rmse"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                break

    return best_epoch, best_val_rmse, hist


def run_cross_validation(
    cfg: CrossValidationConfig, seed=3407, device="auto", seq_len=328
):
    set_random_seed(seed)
    device = get_device(device)
    ensure_dir(cfg.checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model_dir)
    _, ds_train, ds_eval = _build_processor_and_datasets(
        cfg, tokenizer, seq_len
    )

    rec = CVRecoder()
    for cv, fold, train_idx, val_idx, test_idx in make_cross_validation_splits(
        len(ds_eval), n_splits=cfg.n_splits, cv_rounds=cfg.cv_rounds
    ):
        train_loader = DataLoader(
            Subset(ds_train, train_idx),
            batch_size=cfg.batch_train,
            shuffle=True,
            num_workers=cfg.num_workers,
        )
        val_loader = DataLoader(
            Subset(ds_eval, val_idx),
            batch_size=cfg.batch_eval,
            shuffle=False,
            num_workers=cfg.num_workers,
        )
        test_loader = DataLoader(
            Subset(ds_eval, test_idx),
            batch_size=cfg.batch_eval,
            shuffle=False,
            num_workers=cfg.num_workers,
        )

        model = _new_model(cfg.pretrained_model_dir, device)
        best_path = os.path.join(
            cfg.checkpoint_dir, f"cv{cv}_fold{fold}_best.pt"
        )
        best_epoch, best_val_rmse, _ = _train_one_split(
            model,
            train_loader,
            val_loader,
            cfg,
            device,
            best_path,
            f"Fold {fold + 1}",
        )
        model.load_state_dict(torch.load(best_path, map_location=device))
        test_reg, test_clf = evaluate_both(
            model,
            test_loader,
            device,
            threshold=cfg.classification_threshold,
        )
        rec.add(
            cv,
            fold,
            best_epoch,
            best_val_rmse,
            test_reg,
            test_clf,
            cfg.classification_threshold,
        )

    rec.save(cfg.result_csv)


def run_backtest_experiment(
    cfg: BacktestConfig, seed=3407, device="auto", seq_len=328
):
 
    set_random_seed(seed)
    device = get_device(device)
    ensure_dir(cfg.checkpoint_dir)

    
    first_window = f"{cfg.years[0]}_{cfg.hemispheres[0]}"
    first_pretrained_dir = os.path.join(
        cfg.pretrained_model_root, first_window
    )
    tokenizer = AutoTokenizer.from_pretrained(first_pretrained_dir)
    _, ds_train, ds_eval = _build_processor_and_datasets(
        cfg, tokenizer, seq_len
    )

    recorder = BacktestRecorder()
    for split_id, (
        window,
        train_idx,
        val_idx,
        test_idx,
    ) in enumerate(
        split_fludataset_for_backtest(
            ds_eval,
            years=cfg.years,
            hemispheres=cfg.hemispheres,
            val_ratio=cfg.val_ratio,
            random_state=seed,
            descending=cfg.descending,
        ),
        start=1,
    ):
        pretrained_dir = os.path.join(
            cfg.pretrained_model_root, window.window_label
        )
        if not os.path.isdir(pretrained_dir):
            raise FileNotFoundError(
                "Missing independently pretrained model for "
                f"{window.window_label}: {pretrained_dir}"
            )

        window_tokenizer = AutoTokenizer.from_pretrained(pretrained_dir)
         
        _, ds_train_window, ds_eval_window = _build_processor_and_datasets(
            cfg, window_tokenizer, seq_len
        )

        train_loader = DataLoader(
            Subset(ds_train_window, train_idx),
            batch_size=cfg.batch_train,
            shuffle=True,
            num_workers=cfg.num_workers,
        )
        val_loader = DataLoader(
            Subset(ds_eval_window, val_idx),
            batch_size=cfg.batch_eval,
            shuffle=False,
            num_workers=cfg.num_workers,
        )
        test_loader = DataLoader(
            Subset(ds_eval_window, test_idx),
            batch_size=cfg.batch_eval,
            shuffle=False,
            num_workers=cfg.num_workers,
        )

        model = _new_model(pretrained_dir, device)
        best_path = os.path.join(
            cfg.checkpoint_dir, f"{window.window_label}_best.pt"
        )
        best_epoch, best_val_rmse, _ = _train_one_split(
            model,
            train_loader,
            val_loader,
            cfg,
            device,
            best_path,
            window.window_label,
        )

        model.load_state_dict(torch.load(best_path, map_location=device))
        test_reg, test_clf = evaluate_both(
            model,
            test_loader,
            device,
            threshold=cfg.classification_threshold,
        )
        recorder.add_window(
            split_id=split_id,
            window_label=window.window_label,
            year=window.year,
            hemisphere=window.hemisphere,
            cutoff=window.cutoff.isoformat(),
            test_end_exclusive=window.test_end_exclusive.isoformat(),
            train_count=len(train_idx),
            val_count=len(val_idx),
            test_count=len(test_idx),
            best_epoch=best_epoch,
            val_rmse=best_val_rmse,
            test_reg=test_reg,
            test_clf=test_clf,
        )
        recorder.save(cfg.result_csv)

    df = pd.DataFrame(recorder.rows)
    if not df.empty:
        print("\nRetrospective time-split summary:")
        for key in [
            "test_rmse",
            "test_mae",
            "test_r2",
            "test_acc",
            "test_auc",
            "test_auprc",
            "test_f1",
            "test_precision",
            "test_recall",
        ]:
            print(f"{key}: {df[key].mean():.4f} ± {df[key].std():.4f}")
