import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForSequenceClassification

from models.model import CrossEncoder
from utils.data_processing import FLUDataset, H3N2DataProcessor
from utils.metrics import evaluate_both
from configs.config import BacktestConfig, CrossValidationConfig
from utils.recorders import BacktestRecorder, CVRecoder
from train_and_eval.training_strategies import make_cross_validation_splits, split_fludataset_for_backtest
from utils.utils import ensure_dir, get_device, set_random_seed



def _build_processor_and_datasets(cfg, tokenizer, seq_len):
    processor = H3N2DataProcessor(
        ha_path=cfg.ha_path,
        hi_path=cfg.hi_path,
        bin_threshold=cfg.classification_threshold,
        deduplicate=cfg.deduplicate,
    )
    ds_eval = FLUDataset(processor, tokenizer=tokenizer, seq_len=seq_len, mode="test")
    ds_train = FLUDataset(processor, tokenizer=tokenizer, seq_len=seq_len, mode="train")
    return processor, ds_train, ds_eval


def _new_model(pretrained_model_dir, device):
    base = EsmForSequenceClassification.from_pretrained(pretrained_model_dir, num_labels=1).to(device)
    return CrossEncoder(base).to(device)


def _train_one_split(model, train_loader, val_loader, cfg, device, best_path, tag):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )
    criterion = nn.MSELoss()
    best_val_rmse = float("inf")
    no_improve = 0
    best_epoch = 0
    hist = {"mse": [], "rmse": [], "mae": [], "r2": []}

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        with tqdm(train_loader, desc=f" {tag} Epoch {epoch}/{cfg.num_epochs}", unit="batch") as t:
            for input_ids, attention_mask, y_dist, _y_cls in t:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                y_dist = y_dist.to(device).float().unsqueeze(-1)
                preds = model(input_ids, attention_mask)
                loss = criterion(preds, y_dist)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                t.set_postfix(loss=f"{loss.item():.4f}")

        val_reg, val_clf = evaluate_both(model, val_loader, device, threshold=cfg.classification_threshold)
        scheduler.step(val_reg["rmse"])
        hist["mse"].append(val_reg["mse"]); hist["rmse"].append(val_reg["rmse"])
        hist["mae"].append(val_reg["mae"]); hist["r2"].append(val_reg["r2"])
        print(f" Val (Reg) MSE {val_reg['mse']:.4f} | RMSE {val_reg['rmse']:.4f} | MAE {val_reg['mae']:.4f} | R2 {val_reg['r2']:.4f}")
        print(f" Val (Cls thr={cfg.classification_threshold}) ACC {val_clf['acc']:.4f} | AUC {val_clf['auc']:.4f} | AUPRC {val_clf['auprc']:.4f} | F1 {val_clf['f1']:.4f} | P {val_clf['precision']:.4f} | R {val_clf['recall']:.4f}")

        if val_reg["rmse"] + 1e-9 < best_val_rmse:
            best_val_rmse = val_reg["rmse"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
            print(f" Saved best model ({tag}, val_RMSE: {best_val_rmse:.4f})")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f" Early stopping at epoch {epoch}")
                break
    return best_epoch, best_val_rmse, hist


def run_cross_validation(cfg: CrossValidationConfig, seed=3407, device="auto", seq_len=328):
    set_random_seed(seed)
    device = get_device(device)
    ensure_dir(cfg.checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model_dir)
    _, ds_train, ds_eval = _build_processor_and_datasets(cfg, tokenizer, seq_len)
    rec = CVRecoder()

    for cv, fold, train_idx, val_idx, test_idx in make_cross_validation_splits(
        len(ds_eval), n_splits=cfg.n_splits, cv_rounds=cfg.cv_rounds
    ):

        train_loader = DataLoader(Subset(ds_train, train_idx), batch_size=cfg.batch_train, shuffle=True, num_workers=cfg.num_workers)
        val_loader = DataLoader(Subset(ds_eval, val_idx), batch_size=cfg.batch_eval, shuffle=False, num_workers=cfg.num_workers)
        test_loader = DataLoader(Subset(ds_eval, test_idx), batch_size=cfg.batch_eval, shuffle=False, num_workers=cfg.num_workers)

        model = _new_model(cfg.pretrained_model_dir, device)
        best_path = os.path.join(cfg.checkpoint_dir, f"cv{cv}_fold{fold}_best.pt")
        best_epoch, best_val_rmse, hist = _train_one_split(model, train_loader, val_loader, cfg, device, best_path, f"Fold {fold + 1}")
        

        print("\n Evaluating on Test Set...")
        model.load_state_dict(torch.load(best_path, map_location=device))
        test_reg, test_clf = evaluate_both(model, test_loader, device, threshold=cfg.classification_threshold)
        print(f" Test (Reg) - MSE {test_reg['mse']:.4f} | RMSE {test_reg['rmse']:.4f} | MAE {test_reg['mae']:.4f} | R2 {test_reg['r2']:.4f}")
        print(f" Test (Cls thr={cfg.classification_threshold}) - ACC {test_clf['acc']:.4f} | AUC {test_clf['auc']:.4f} | AUPRC {test_clf['auprc']:.4f} | F1 {test_clf['f1']:.4f} | P {test_clf['precision']:.4f} | R {test_clf['recall']:.4f}")
        rec.add(cv, fold, best_epoch, best_val_rmse, test_reg, test_clf, cfg.classification_threshold)

    rec.save(cfg.result_csv)
    df = pd.DataFrame(rec.rows)
    if not df.empty:
        print("\n Final Cross-Validation Results:")
        for k in ["test_rmse", "test_mae", "test_r2", "test_acc", "test_auc", "test_auprc", "test_f1", "test_precision", "test_recall"]:
            if k in df.columns:
                print(f"{k}: {df[k].mean():.4f} ± {df[k].std():.4f}")


def run_backtest_experiment(cfg: BacktestConfig, seed=3407, device="auto", seq_len=328):
    set_random_seed(seed)
    device = get_device(device)
    ensure_dir(cfg.checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model_dir)
    _, ds_train, ds_eval = _build_processor_and_datasets(cfg, tokenizer, seq_len)

    train_idx_list, val_idx_list, test_idx_list, test_year_list = split_fludataset_for_backtest(
        ds_eval,
        train_year_num=cfg.train_year_num,
        test_year_num=cfg.test_year_num,
        val_ratio=cfg.val_ratio,
        min_test_start_year=cfg.min_test_start_year,
        descending=cfg.descending,
    )
    recorder = BacktestRecorder()

    for split_id, (train_idx, val_idx, test_idx, test_start_year) in enumerate(
        zip(train_idx_list, val_idx_list, test_idx_list, test_year_list), start=1
    ):
        
        train_loader = DataLoader(Subset(ds_train, train_idx), batch_size=cfg.batch_train, shuffle=True, num_workers=cfg.num_workers)
        val_loader = DataLoader(Subset(ds_eval, val_idx), batch_size=cfg.batch_eval, shuffle=False, num_workers=cfg.num_workers)
        test_loader = DataLoader(Subset(ds_eval, test_idx), batch_size=cfg.batch_eval, shuffle=False, num_workers=cfg.num_workers)

        model = _new_model(cfg.pretrained_model_dir, device)
        best_path = os.path.join(cfg.checkpoint_dir, f"split{split_id}_best.pt")
        best_epoch, best_val_rmse, hist = _train_one_split(model, train_loader, val_loader, cfg, device, best_path, f"Split {split_id}")
        
        print("\n Evaluating on Test Set...")
        model.load_state_dict(torch.load(best_path, map_location=device))
        test_reg, test_clf = evaluate_both(model, test_loader, device, threshold=cfg.classification_threshold)
        print(f" Test (Reg) - MSE {test_reg['mse']:.4f} | RMSE {test_reg['rmse']:.4f} | MAE {test_reg['mae']:.4f} | R2 {test_reg['r2']:.4f}")
        print(f" Test (Cls thr={cfg.classification_threshold}) - ACC {test_clf['acc']:.4f} | AUC {test_clf['auc']:.4f} | AUPRC {test_clf['auprc']:.4f} | F1 {test_clf['f1']:.4f} | P {test_clf['precision']:.4f} | R {test_clf['recall']:.4f}")
        recorder.add(split_id, test_start_year, best_epoch, best_val_rmse, test_reg, test_clf)

    recorder.save(cfg.result_csv)
    df = pd.DataFrame(recorder.rows)
    print("\n Backtest Summary:")
    for k in ["test_rmse", "test_mae", "test_r2", "test_acc", "test_auc", "test_auprc", "test_f1", "test_precision", "test_recall"]:
        if k in df.columns:
            print(f"{k}: {df[k].mean():.4f} ± {df[k].std():.4f}")
    
