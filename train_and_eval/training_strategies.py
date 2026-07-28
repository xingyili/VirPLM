from dataclasses import dataclass
from datetime import date
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import KFold, train_test_split


@dataclass(frozen=True)
class BacktestWindow:
 

    window_label: str
    year: int
    hemisphere: str
    cutoff: date
    test_end_exclusive: date


def make_cross_validation_splits(dataset_size, n_splits=5, cv_rounds=1):
 
    idx = np.arange(dataset_size)
    for cv in range(cv_rounds):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=cv)
        for fold, (train_val_idx, test_idx) in enumerate(kf.split(idx)):
            train_val_idx = np.asarray(train_val_idx)
            test_idx = np.asarray(test_idx)
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=0.25,
                random_state=cv,
                shuffle=True,
            )
            yield cv, fold, train_idx, val_idx, test_idx


def build_retrospective_windows(
    years: Sequence[int] = tuple(range(2012, 2025)),
    hemispheres: Sequence[str] = ("NH", "SH"),
) -> List[BacktestWindow]:
  
    windows: List[BacktestWindow] = []
    for year in years:
        for hemisphere in hemispheres:
            if hemisphere == "NH":
                cutoff = date(year - 1, 9, 1)
                test_end = date(year, 2, 1)
            elif hemisphere == "SH":
                cutoff = date(year, 2, 1)
                test_end = date(year, 9, 1)
            else:
                raise ValueError(f"Unsupported hemisphere: {hemisphere!r}")

            windows.append(
                BacktestWindow(
                    window_label=f"{year}_{hemisphere}",
                    year=int(year),
                    hemisphere=hemisphere,
                    cutoff=cutoff,
                    test_end_exclusive=test_end,
                )
            )
    return windows


def _validate_dataset_dates(dataset) -> None:
    required = ("at_dates", "sr_dates")
    missing = [name for name in required if not hasattr(dataset, name)]
    if missing:
        raise AttributeError(
            "The retrospective split requires exact collection dates. "
            f"Dataset is missing attributes: {missing}. "
            "Use the updated H3N2DataProcessor/FLUDataset."
        )
    if len(dataset.at_dates) != len(dataset) or len(dataset.sr_dates) != len(dataset):
        raise ValueError("Dataset date arrays do not match dataset length.")


def split_fludataset_for_backtest(
    dataset,
    years: Sequence[int] = tuple(range(2012, 2025)),
    hemispheres: Sequence[str] = ("NH", "SH"),
    val_ratio: float = 0.2,
    random_state: int = 3407,
    descending: bool = False,
) -> Iterator[Tuple[BacktestWindow, List[int], List[int], List[int]]]:
     
    _validate_dataset_dates(dataset)
    windows = build_retrospective_windows(years=years, hemispheres=hemispheres)
    if descending:
        windows = list(reversed(windows))

    for window in windows:
        train_all_idx = [
            i
            for i, (at_date, sr_date) in enumerate(zip(dataset.at_dates, dataset.sr_dates))
            if at_date < window.cutoff and sr_date < window.cutoff
        ]
        test_idx = [
            i
            for i, (at_date, sr_date) in enumerate(zip(dataset.at_dates, dataset.sr_dates))
            if window.cutoff <= at_date < window.test_end_exclusive
            and sr_date < window.cutoff
        ]

        if len(train_all_idx) < 2:
            print(
                f"[Skip {window.window_label}] Too few historical pairs: "
                f"{len(train_all_idx)}"
            )
            continue
        if not test_idx:
            print(f"[Skip {window.window_label}] No eligible test pairs.")
            continue

        train_idx, val_idx = train_test_split(
            train_all_idx,
            test_size=val_ratio,
            random_state=random_state,
            shuffle=True,
        )

        # Defensive leakage checks.
        for idx in list(train_idx) + list(val_idx):
            assert dataset.at_dates[idx] < window.cutoff
            assert dataset.sr_dates[idx] < window.cutoff
        for idx in test_idx:
            assert window.cutoff <= dataset.at_dates[idx] < window.test_end_exclusive
            assert dataset.sr_dates[idx] < window.cutoff

        print(
            f"[{window.window_label}] cutoff={window.cutoff.isoformat()}, "
            f"test=[{window.cutoff.isoformat()}, "
            f"{window.test_end_exclusive.isoformat()}), "
            f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
        )
        yield window, list(train_idx), list(val_idx), list(test_idx)
