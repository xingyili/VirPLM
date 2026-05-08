import argparse
from dataclasses import fields

from configs.config import BacktestConfig, CrossValidationConfig
from train_and_eval.fine_tuning import run_backtest_experiment, run_cross_validation
from train_and_eval.pretrain import run_pretraining
from utils.utils import load_yaml


def _apply_overrides(section_cfg, args):

    overrides = {
        "pretrained_model_dir": args.pretrained_model_dir,
        "ha_path": args.ha_path,
        "hi_path": args.hi_path,
        "result_csv": args.result_csv,
        "checkpoint_dir": args.checkpoint_dir,
        "num_epochs": args.num_epochs,
        "classification_threshold": args.classification_threshold,
    }
    for k, v in overrides.items():
        if v is not None and k in section_cfg:
            section_cfg[k] = v
    return section_cfg


def _dataclass_from_dict(cls, cfg_dict):
    allowed = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in cfg_dict.items() if k in allowed})


def build_parser():
    parser = argparse.ArgumentParser(description="VirPLM: pretraining, cross-validation, and retrospective backtest")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to yaml config file")
    parser.add_argument("--mode", choices=["pretrain", "cv", "backtest"], required=True, help="Pipeline to run")

    parser.add_argument("--pretrained-model-dir", dest="pretrained_model_dir", default=None)
    parser.add_argument("--ha-path", default=None)
    parser.add_argument("--hi-path", default=None)
    parser.add_argument("--result-csv", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--classification-threshold", type=float, default=None)

    parser.add_argument("--base-model-name", default=None)
    parser.add_argument("--fasta-paths", nargs="+", default=None)
    parser.add_argument("--output-dir", default=None)

    parser.add_argument("--train-year-num", type=int, default=None)
    parser.add_argument("--test-year-num", type=int, default=None)
    parser.add_argument("--min-test-start-year", type=int, default=None)
    return parser


def main():
    args = build_parser().parse_args()
    cfg = load_yaml(args.config)
    seed = cfg.get("seed", 3407)
    device = cfg.get("device", "auto")
    seq_len = cfg.get("seq_len", 328)

    if args.mode == "pretrain":
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
        ft_cfg = _apply_overrides(dict(cfg["cross_validation"]), args)
        run_cross_validation(_dataclass_from_dict(CrossValidationConfig, ft_cfg), seed=seed, device=device, seq_len=seq_len)
        return

    if args.mode == "backtest":
        bt_cfg = _apply_overrides(dict(cfg["backtest"]), args)
        if args.train_year_num is not None:
            bt_cfg["train_year_num"] = args.train_year_num
        if args.test_year_num is not None:
            bt_cfg["test_year_num"] = args.test_year_num
        if args.min_test_start_year is not None:
            bt_cfg["min_test_start_year"] = args.min_test_start_year
        run_backtest_experiment(_dataclass_from_dict(BacktestConfig, bt_cfg), seed=seed, device=device, seq_len=seq_len)


if __name__ == "__main__":
    main()
