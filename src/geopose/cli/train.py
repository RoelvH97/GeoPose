"""Frozen training entry points for the two published GeoPose stages."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from ..init.data import CTAPoseDataModule
from ..refine.data import SyntheticRefineDataModule
from ..init import ResNetPose
from ..refine import RefinePoseModule


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPOSITORY_ROOT / "configs"
SPLIT_FILE = REPOSITORY_ROOT / "assets" / "isles_split_v1.json"


def _existing_directory(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"Directory does not exist: {path}")
    return path


def _existing_file(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File does not exist: {path}")
    return path


def _devices(value: str):
    return int(value) if value.isdigit() else value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the frozen GeoPose-Init or GeoPose-Refine publication recipe."
    )
    parser.add_argument("stage", choices=("init", "refine"))
    parser.add_argument("--data-root", required=True, type=_existing_directory)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--init-checkpoint",
        type=_existing_file,
        help="Required for Refine: GeoPose-Init checkpoint used to warm-start its encoder.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--accelerator")
    parser.add_argument("--devices", type=_devices)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--epoch-length", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a two-sample, one-epoch integration check without changing model/loss settings.",
    )
    parser.add_argument("--wandb", action="store_true", help="Also log to Weights & Biases.")
    return parser


def load_training_contract(stage: str) -> DictConfig:
    cfg = OmegaConf.load(CONFIG_DIR / f"{stage}.yaml")
    if cfg.release_contract.version != f"geopose-{stage}-v1":
        raise RuntimeError(f"Unexpected {stage} training contract")
    return cfg


def _configure(args: argparse.Namespace) -> DictConfig:
    cfg = load_training_contract(args.stage)
    cfg.data.data_root = str(args.data_root)

    cfg.data.split_file = str(SPLIT_FILE)
    cfg.trainer.log_dir = str(args.output_dir.resolve())

    if args.accelerator is not None:
        cfg.trainer.accelerator = args.accelerator
    if args.devices is not None:
        cfg.trainer.devices = args.devices
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers
    if args.max_epochs is not None:
        cfg.trainer.max_epochs = args.max_epochs
    if args.epoch_length is not None:
        cfg.data.epoch_len = args.epoch_length
    if args.batch_size is not None:
        if args.stage == "init":
            cfg.data.batch_size = args.batch_size
        else:
            cfg.refine.batch_size = args.batch_size

    if args.stage == "refine":
        if args.init_checkpoint is None:
            raise ValueError("--init-checkpoint is required for stage=refine")
        cfg.model.init_encoder_ckpt = str(args.init_checkpoint)
    elif args.init_checkpoint is not None:
        raise ValueError("--init-checkpoint is only valid for stage=refine")

    if args.smoke:
        cfg.trainer.max_epochs = 1
        cfg.data.epoch_len = 2
        cfg.data.max_subjects = 3
        cfg.data.num_workers = 0
        if args.stage == "init":
            cfg.data.batch_size = 2
            cfg.trainer.accumulate_grad_batches = 1
        else:
            cfg.refine.batch_size = 2
            cfg.trainer.accumulate_grad_batches = 1

    OmegaConf.resolve(cfg)
    return cfg


def _loggers(cfg: DictConfig, use_wandb: bool):
    csv = CSVLogger(save_dir=cfg.trainer.log_dir, name="metrics", version="")
    if not use_wandb:
        return csv
    online = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity or None,
        name=cfg.wandb.name or None,
        tags=list(cfg.wandb.tags),
        config=OmegaConf.to_container(cfg, resolve=True),
        save_dir=cfg.trainer.log_dir,
    )
    return [csv, online]


def run(args: argparse.Namespace) -> None:
    cfg = _configure(args)
    output_dir = Path(cfg.trainer.log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml")

    pl.seed_everything(args.seed, workers=True)
    if args.stage == "init":
        datamodule = CTAPoseDataModule(cfg.data)
        model = ResNetPose(cfg.model)
        checkpoint = ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            filename="{epoch}-{val_loss:.4f}",
        )
    else:
        datamodule = SyntheticRefineDataModule(cfg)
        model = RefinePoseModule(cfg)

        pl.seed_everything(args.seed, workers=True)
        checkpoint = ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            filename="{epoch}-{val_loss:.4f}",
        )

    smoke_limits = (
        {
            "limit_train_batches": 1,
            "limit_val_batches": 1,
            "num_sanity_val_steps": 0,
        }
        if args.smoke
        else {}
    )
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="epoch")],
        logger=_loggers(cfg, args.wandb),
        default_root_dir=cfg.trainer.log_dir,
        **smoke_limits,
    )
    trainer.fit(model, datamodule=datamodule)


def main(argv: list[str] | None = None) -> None:
    run(_parser().parse_args(argv))


if __name__ == "__main__":
    main()


