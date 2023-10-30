import traceback

from omegaconf import OmegaConf
from torch import manual_seed
from pytorch_lightning import Trainer

from utils import (
    get_curr_time_w_random_shift,
    maybe_save_checkpoint,
    init_log_directory,
    get_logger,
    get_callbacks,
)
from vggishishmodule import VGGishishModule
from datamodule import GreatesHitDataModule


def train(cfg: OmegaConf):
    print(cfg.start_time, cfg.get("exp_name", ""))
    manual_seed(cfg.get("seed", 666))

    log_dir, ckpt_dir = init_log_directory(cfg.start_time, cfg.log_dir)

    model = VGGishishModule(**cfg.model)
    datamodule = GreatesHitDataModule(**cfg.dataloader)
    trainer = Trainer(
        callbacks=get_callbacks(ckpt_dir),
        logger=get_logger(log_dir, name=cfg.trainer.get("exp_name", "")),
        **cfg.trainer
    )

    try:
        trainer.fit(model=model, datamodule=datamodule)
    except BaseException as e:
        print(e)
        traceback.print_exc()
        maybe_save_checkpoint(trainer)


if __name__ == "__main__":
    args = OmegaConf.from_cli()
    config = OmegaConf.load(args.get("config", "./configs/vggishish.yaml"))
    config = OmegaConf.merge(config, args)
    if "start_time" not in config or config.start_time is None:
        config.start_time = get_curr_time_w_random_shift()
    OmegaConf.resolve(
        config
    )  # things like "${model.size}" in cfg will be resolved into values
    train(config)
