import pandas as pd


class CVRecoder:
    def __init__(self):
        self.rows = []

    def add(self, cv, fold, best_epoch, val_rmse, test_reg, test_clf, thr):
        self.rows.append({
            "cv_round": cv + 1,
            "fold": fold + 1,
            "best_epoch": best_epoch,
            "val_rmse": val_rmse,
            "test_mse": test_reg["mse"],
            "test_rmse": test_reg["rmse"],
            "test_mae": test_reg["mae"],
            "test_r2": test_reg["r2"],
            "threshold": thr,
            "test_acc": test_clf["acc"],
            "test_auc": test_clf["auc"],
            "test_auprc": test_clf["auprc"],
            "test_f1": test_clf["f1"],
            "test_precision": test_clf["precision"],
            "test_recall": test_clf["recall"],
        })

    def save(self, path="CV_results.csv"):
        pd.DataFrame(self.rows).to_csv(path, index=False)


class BacktestRecorder:
    def __init__(self):
        self.rows = []

    def add(self, split_id, test_start_year, best_epoch, val_rmse, test_reg, test_clf):
        self.rows.append({
            "split_id": split_id,
            "test_start_year": test_start_year,
            "best_epoch": best_epoch,
            "val_rmse": val_rmse,
            "test_mse": test_reg["mse"],
            "test_rmse": test_reg["rmse"],
            "test_mae": test_reg["mae"],
            "test_r2": test_reg["r2"],
            "test_acc": test_clf["acc"],
            "test_auc": test_clf["auc"],
            "test_auprc": test_clf["auprc"],
            "test_f1": test_clf["f1"],
            "test_precision": test_clf["precision"],
            "test_recall": test_clf["recall"],
        })

    def save(self, path="Backtest_results.csv"):
        pd.DataFrame(self.rows).to_csv(path, index=False)
