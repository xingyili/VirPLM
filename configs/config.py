from dataclasses import dataclass
from typing import Optional


@dataclass
class CrossValidationConfig:
    pretrained_model_dir: str
    ha_path: str
    hi_path: str
    extra_ha_path: Optional[str] = None
    extra_hi_path: Optional[str] = None
    num_epochs: int = 50
    classification_threshold: float = 2.0
    batch_train: int = 8
    batch_eval: int = 16
    lr: float = 5e-5
    patience: int = 10
    num_workers: int = 4
    checkpoint_dir: str = "checkpoints_cv"
    result_csv: str = "cv_results.csv"
    cv_rounds: int = 1
    n_splits: int = 5
    skip_folds_before: int = 0
    deduplicate: bool = True


@dataclass
class BacktestConfig:
    pretrained_model_dir: str
    ha_path: str
    hi_path: str
    extra_ha_path: Optional[str] = None
    extra_hi_path: Optional[str] = None
    num_epochs: int = 50
    classification_threshold: float = 2.0
    batch_train: int = 8
    batch_eval: int = 16
    lr: float = 5e-5
    patience: int = 10
    num_workers: int = 4
    checkpoint_dir: str = "checkpoints_backtest"
    result_csv: str = "backtest_results.csv"
    train_year_num: int = 7
    test_year_num: int = 1
    val_ratio: float = 0.2
    min_test_start_year: Optional[int] = 2012
    descending: bool = True
    deduplicate: bool = False
