import argparse
from dataclasses import fields
from pathlib import Path

from configs.config import BacktestConfig, CrossValidationConfig
from utils.utils import load_yaml


def _apply_overrides(section_cfg, args, keys):
 
    for key in keys:
        value = getattr(args, key, None)
        if value is not None:
            section_cfg[key] = value
    return section_cfg


def _dataclass_from_dict(cls, cfg_dict):
    allowed = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in cfg_dict.items() if k in allowed})


def _resolve_project_path(value):
 
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((Path(__file__).resolve().parent / path).resolve())


def _normalize_backtest_paths(bt_cfg):
    for key in (
        "pretrained_model_root",
        "ha_path",
        "hi_path",
        "metadata_path",
        "extra_ha_path",
        "extra_hi_path",
        "checkpoint_dir",
        "result_csv",
    ):
        if bt_cfg.get(key) is not None:
            bt_cfg[key] = _resolve_project_path(bt_cfg[key])
    return bt_cfg


def _validate_backtest_inputs(bt_cfg):
    required_files = {
        "ha_path": bt_cfg.ha_path,
        "hi_path": bt_cfg.hi_path,
        "metadata_path": bt_cfg.metadata_path,
    }
    missing_files = [f"{key}={value}" for key, value in required_files.items() if not Path(value).is_file()]
    if missing_files:
        raise FileNotFoundError(
            "Missing retrospective input file(s): " + ", ".join(missing_files)
        )

    missing_models = []
    for year in bt_cfg.years:
        for hemisphere in bt_cfg.hemispheres:
            model_dir = Path(bt_cfg.pretrained_model_root) / f"{year}_{hemisphere}"
            if not model_dir.is_dir():
                missing_models.append(str(model_dir))
    if missing_models:
        shown = ", ".join(missing_models[:4])
        extra = "" if len(missing_models) <= 4 else f" ... (+{len(missing_models) - 4} more)"
        raise FileNotFoundError(
            "Missing independently pretrained retrospective model directory/directories: "
            f"{shown}{extra}. Each requested window needs a directory named "
            "<year>_<NH|SH> under --pretrained-model-root."
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description="VirPLM: pretraining, cross-validation, and retrospective backtest"
    )
    parser.add_argument(
        "--config", default="configs/default.yaml", help="Path to YAML config file"
    )
    parser.add_argument(
        "--mode",
        choices=["pretrain", "cv", "backtest"],
        required=True,
        help="Pipeline to run",
    )

 
    parser.add_argument("--pretrained-model-dir", default=None)
    parser.add_argument("--ha-path", default=None)
    parser.add_argument("--hi-path", default=None)
    parser.add_argument("--result-csv", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--classification-threshold", type=float, default=None)

 
    parser.add_argument("--base-model-name", default=None)
    parser.add_argument("--fasta-paths", nargs="+", default=None)
    parser.add_argument("--output-dir", default=None)

 
    parser.add_argument(
        "--pretrained-model-root",
        default=None,
        help=(
            "Root directory containing independently pretrained models such as "
            "2020_NH and 2020_SH"
        ),
    )
    parser.add_argument(
        "--metadata-path",
        default=None,
        help="CSV containing header and Collection_Date columns",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=None,
        help="Retrospective vaccine-season years, e.g. --years 2020",
    )
    parser.add_argument(
        "--hemispheres",
        nargs="+",
        choices=["NH", "SH"],
        default=None,
        help="Seasonal windows to evaluate, e.g. --hemispheres NH",
    )
    order_group = parser.add_mutually_exclusive_group()
    order_group.add_argument(
        "--descending", dest="descending", action="store_true", default=None
    )
    order_group.add_argument(
        "--ascending", dest="descending", action="store_false"
    )
    return parser


def main():
    args = build_parser().parse_args()
    cfg = load_yaml(_resolve_project_path(args.config))
    seed = cfg.get("seed", 3407)
    device = cfg.get("device", "auto")
    seq_len = cfg.get("seq_len", 328)

    if args.mode == "pretrain":
        from train_and_eval.pretrain import run_pretraining

        pre_cfg = dict(cfg["pretrain"])
        if args.base_model_name is not None:
            pre_cfg["base_model_name"] = args.base_model_name
        if args.fasta_paths is not None:
            pre_cfg["fasta_paths"] = args.fasta_paths
        if args.output_dir is not None:
            pre_cfg["output_dir"] = args.output_dir
        if args.num_epochs is not None:
            pre_cfg["epochs"] = args.num_epochs
        run_pretraining(pre_cfg, seed=seed, device=device, seq_len=seq_len)
        return

    if args.mode == "cv":
        from train_and_eval.fine_tuning import run_cross_validation

        ft_cfg = _apply_overrides(
            dict(cfg["cross_validation"]),
            args,
            (
                "pretrained_model_dir",
                "ha_path",
                "hi_path",
                "result_csv",
                "checkpoint_dir",
                "num_epochs",
                "classification_threshold",
            ),
        )
        run_cross_validation(
            _dataclass_from_dict(CrossValidationConfig, ft_cfg),
            seed=seed,
            device=device,
            seq_len=seq_len,
        )
        return

    bt_cfg = _apply_overrides(
        dict(cfg["backtest"]),
        args,
        (
            "pretrained_model_root",
            "ha_path",
            "hi_path",
            "metadata_path",
            "result_csv",
            "checkpoint_dir",
            "num_epochs",
            "classification_threshold",
            "years",
            "hemispheres",
            "descending",
        ),
    )

   
    if args.pretrained_model_root is None and args.pretrained_model_dir is not None:
        bt_cfg["pretrained_model_root"] = args.pretrained_model_dir

    bt_cfg = _normalize_backtest_paths(bt_cfg)
    bt_dataclass = _dataclass_from_dict(BacktestConfig, bt_cfg)
    _validate_backtest_inputs(bt_dataclass)

    from train_and_eval.fine_tuning import run_backtest_experiment

    run_backtest_experiment(
        bt_dataclass, seed=seed, device=device, seq_len=seq_len
    )


if __name__ == "__main__":
    main()
