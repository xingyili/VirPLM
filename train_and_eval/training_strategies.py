from typing import List, Optional, Tuple
import numpy as np
from sklearn.model_selection import KFold, train_test_split


def make_cross_validation_splits(dataset_size, n_splits=5, cv_rounds=1):
    idx = np.arange(dataset_size)
    for cv in range(cv_rounds):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=cv)
        for fold, (train_val_idx, test_idx) in enumerate(kf.split(idx)):
            train_val_idx = np.asarray(train_val_idx)
            test_idx = np.asarray(test_idx)
            train_idx, val_idx = train_test_split(
                train_val_idx, test_size=0.25, random_state=cv, shuffle=True
            )
            yield cv, fold, train_idx, val_idx, test_idx


def split_fludataset_for_backtest(
    dataset,
    train_year_num: int = 14,
    test_year_num: int = 2,
    val_ratio: float = 0.2,
    random_state: int = 42,
    min_test_start_year: Optional[int] = None,
    descending: bool = True,
) -> Tuple[List[List[int]], List[List[int]], List[List[int]], List[int]]:
    at_year = dataset.at_years
    sr_year = dataset.sr_years
    valid_years = [y for y in at_year if 1900 < y < 2100]
    if not valid_years:
        raise ValueError("未找到有效年份，请检查HA文件的 year 列。")

    all_years = sorted(list(set(valid_years)))
    print("时间轴 (Years):", all_years)
    total_cv = len(all_years) - train_year_num - test_year_num + 1
    if total_cv <= 0:
        raise ValueError("可滑动的窗口数为 0，请调整 train_year_num / test_year_num。")

    tmp_train_idx_list, tmp_val_idx_list, tmp_test_idx_list, tmp_test_year_list = [], [], [], []
    tmp_train_years_list, tmp_test_years_list = [], []

    for cv in range(total_cv):
        train_years = all_years[0:cv + train_year_num]
        test_years = all_years[cv + train_year_num: cv + train_year_num + test_year_num]
        if not test_years:
            continue
        test_start_year = int(test_years[0])

        train_all_idx = [
            i for i, y in enumerate(at_year)
            if (y in train_years) and (sr_year[i] < test_start_year)
        ]
        test_idx = [
            i for i, y in enumerate(at_year)
            if (y in test_years) and (sr_year[i] < test_start_year)
        ]

        if len(train_all_idx) < 2:
            print(f"[原始 CV {cv + 1}] 训练样本太少，跳过该窗口（test_start_year={test_start_year}）。")
            continue

        train_idx, val_idx = train_test_split(
            train_all_idx,
            test_size=val_ratio,
            random_state=random_state,
            shuffle=True,
        )
        tmp_train_idx_list.append(train_idx)
        tmp_val_idx_list.append(val_idx)
        tmp_test_idx_list.append(test_idx)
        tmp_test_year_list.append(test_start_year)
        tmp_train_years_list.append(train_years)
        tmp_test_years_list.append(test_years)

    if not tmp_test_year_list:
        raise ValueError("没有生成任何有效的时间窗口，请检查年份和参数设置。")

    if min_test_start_year is not None:
        filtered = [
            (tr, va, te, ty, tr_y, te_y)
            for tr, va, te, ty, tr_y, te_y in zip(
                tmp_train_idx_list,
                tmp_val_idx_list,
                tmp_test_idx_list,
                tmp_test_year_list,
                tmp_train_years_list,
                tmp_test_years_list,
            )
            if ty >= min_test_start_year
        ]
        if not filtered:
            raise ValueError(f"没有满足 test_start_year >= {min_test_start_year} 的窗口。")
        tmp_train_idx_list, tmp_val_idx_list, tmp_test_idx_list, tmp_test_year_list, tmp_train_years_list, tmp_test_years_list = map(list, zip(*filtered))

    order = sorted(range(len(tmp_test_year_list)), key=lambda i: tmp_test_year_list[i])
    if descending:
        order = order[::-1]

    train_idx_list = [tmp_train_idx_list[i] for i in order]
    val_idx_list = [tmp_val_idx_list[i] for i in order]
    test_idx_list = [tmp_test_idx_list[i] for i in order]
    test_year_list = [tmp_test_year_list[i] for i in order]
    train_years_list = [tmp_train_years_list[i] for i in order]
    test_years_list = [tmp_test_years_list[i] for i in order]

    print("=== 回顾性时间窗口（按实际训练顺序）===")
    for k, (tr_y, te_y, ty, tr_idx, va_idx, te_idx) in enumerate(
        zip(train_years_list, test_years_list, test_year_list, train_idx_list, val_idx_list, test_idx_list),
        start=1,
    ):
        print(
            f"[Split {k}] 训练年份范围={tr_y[0]}–{tr_y[-1]}，"
            f"测试年份范围={te_y[0]}–{te_y[-1]}，"
            f"测试起始年={ty}，样本: train={len(tr_idx)}, "
            f"val={len(va_idx)}, test={len(te_idx)}"
        )

    return train_idx_list, val_idx_list, test_idx_list, test_year_list
