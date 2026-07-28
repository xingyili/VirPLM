from dataclasses import dataclass, field
from typing import Sequence, Optional


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
    # Root containing independently pretrained models:
    # pretrained_model_root/2012_NH, ..., pretrained_model_root/2024_SH
    pretrained_model_root: str
    ha_path: str
    hi_path: str
    metadata_path: str
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
    val_ratio: float = 0.2
    years: Sequence[int] = field(default_factory=lambda: tuple(range(2012, 2025)))
    hemispheres: Sequence[str] = ("NH", "SH")
    descending: bool = False
    deduplicate: bool = False

