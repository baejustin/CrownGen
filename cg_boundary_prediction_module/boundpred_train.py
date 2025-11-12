
import argparse
import random
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from omegaconf import OmegaConf

from dataset.dentition_data import DentitionDataModule
from model.bound_encoder import BoundEncoder


class LearningRateTrackerCallback(Callback):
    """Callback to log learning rate at the start of each training epoch."""
    
    def on_train_epoch_start(self, trainer, pl_module):
        """Log current learning rate from optimizer."""
        optimizers = trainer.optimizers
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                pl_module.log('current_lr', param_group['lr'])
        return super().on_train_epoch_start(trainer, pl_module)


def setup_callbacks(cfg):

    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.checkpoint.monitor,
        filename=cfg.checkpoint.filename,
        save_top_k=cfg.checkpoint.save_top_k,
        mode=cfg.checkpoint.mode,
        save_last=cfg.checkpoint.save_last
    )
    
    lr_tracker = LearningRateTrackerCallback()
    
    return [checkpoint_callback, lr_tracker]


def setup_trainer(cfg, callbacks):

    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        log_every_n_steps=cfg.training.log_every_n_steps,
        strategy=cfg.trainer.strategy,
        callbacks=callbacks
    )
    return trainer


def main(cfg):
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    
    data_module = DentitionDataModule(cfg)
    data_module.setup()
    
    model = BoundEncoder(cfg)
    
    callbacks = setup_callbacks(cfg)
    
    trainer = setup_trainer(cfg, callbacks)
    
    ckpt_path = cfg.checkpoint.resume_from if cfg.checkpoint.resume_from else None
    if ckpt_path:
        trainer.fit(model, data_module, ckpt_path=ckpt_path)
    else:
        trainer.fit(model, data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default='configs/boundpred_cfg.yaml'
    )
    cli_args = parser.parse_args()

    main(OmegaConf.load(cli_args.config))