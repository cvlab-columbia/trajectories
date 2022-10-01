import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint  # , LearningRateMonitor
from pytorch_lightning.utilities.seed import seed_everything

import data as data
import models
import utils
from utils.optimizer import LearningRateMonitorSlowFast as LearningRateMonitor


@hydra.main(config_path="configs", config_name="default")
def train(config: omegaconf.DictConfig):
    torch.autograd.detect_anomaly()
    seed_everything(seed=config.seed, workers=True)  # For reproducibility purposes
    logger = utils.Logger(**config.wandb)

    datamodule = data.get_datamodule(**config.dataset)

    model_class, model_cfg = models.get_model(**config.model)

    # --------- Manage resume or create new model ------- #
    checkpoint_path = utils.get_checkpoint_path(logger, config.resume) if config.resume.load_any else None

    if config.resume.load_all:
        assert config.resume.id is not None
        if config.resume.check_config:
            utils.check_same_config(config, logger)  # Before logging the potentially incorrect config

    if config.resume.load_model:
        model = utils.load_model(model_class, checkpoint_path, model_cfg)
    else:
        model = model_class(**model_cfg)
    # -------------------------------------------------- #

    if config.wandb.save:
        logger.log_model_summary(model)
        logger.log_config(config)
        logger.log_code()

    checkpoint_callback = ModelCheckpoint(**config.checkpoint)
    lr_monitor = LearningRateMonitor()

    trainer = pl.Trainer(
        resume_from_checkpoint=checkpoint_path if (config.resume.load_all or config.resume.load_state) else None,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        # Manual mode backward does not support automatic gradient clipping
        gradient_clip_val=config.model.optim_params.grad_clip_val,
        gradient_clip_algorithm=config.model.optim_params.grad_clip_strategy,
        **config.trainer
    )

    if config.setting == 'train':
        trainer.fit(model, datamodule)
    elif config.setting == 'test':
        # Test standard accuracies defined in the Trainer metrics
        trainer.test(model, datamodule, verbose=config.verbose)
    elif config.setting == 'predict':
        # Other predictions, for which the standard loop is not enough
        trainer.predict(model, datamodule, return_predictions=False)


if __name__ == "__main__":
    train()
