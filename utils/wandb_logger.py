# From Tim
from __future__ import annotations

__all__ = ["Logger"]

import os
import pathlib
from typing import Optional

import omegaconf
import torch
import torchvision.utils as utils
import wandb
from pytorch_lightning import LightningModule
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only


def is_rank_zero():
    rank = int(os.environ.get("LOCAL_RANK", 0))
    return rank == 0


class Logger(WandbLogger):
    def __init__(
            self,
            *,
            name: str,
            project: str,
            entity: str,
            group: Optional[str] = None,
            offline: bool = False,
            save_dir: str = None,
            id: str = None,
            save: bool = True
    ):
        if save_dir is None:
            save_dir = str(pathlib.Path(__file__).parents[1])

        if not save:
            offline = True  # Does not log the result to the server. Still creates a folder locally

        super().__init__(
            name=name,
            save_dir=save_dir,
            offline=offline,
            project=project,
            log_model=False,
            entity=entity,
            group=group,
            id=id,
        )

    @rank_zero_only
    def log_config(self, config: omegaconf.DictConfig):
        """
        Save hydra config in the logs folder
        """

        if is_rank_zero():
            hydra_config = omegaconf.OmegaConf.to_yaml(config)

            filename = "hydra_config.yaml"
            self.experiment.save(filename)

            path = os.path.join(self.experiment.dir, filename)
            with open(path, "w") as file:
                print(hydra_config, file=file)

            params = omegaconf.OmegaConf.to_container(config)
            assert isinstance(params, dict)

            # The wandb and resume parameters can be different across runs, even for load_all.
            params.pop("wandb", None)
            params.pop("resume", None)

            self.experiment.config.update(params, allow_val_change=True)

    @rank_zero_only
    def log_model_summary(self, model: LightningModule):

        if is_rank_zero():
            summary = ModelSummary(model)  # , mode='full')

            filename = "model_summary.txt"
            self.experiment.save(filename)

            path = os.path.join(self.experiment.dir, filename)
            with open(path, "w") as file:
                print(summary, file=file)

    @rank_zero_only
    def log_code(self):
        """
        For now, just saving the information about the last git commit
        """
        # repo = git.Repo(path=os.path.dirname(os.path.realpath(__file__)), search_parent_directories=True)
        # sha = repo.head.object.hexsha
        # message_last_commit = repo.head.commit.message

        """
        We don't do anything because this will not work if the git is controlled locally and the code is deployed in a
        machine without an updated git
        """
        return

    @rank_zero_only
    def log_hist(self, name, x):
        if is_rank_zero() and x is not None:
            wandb.log({name: wandb.Histogram(x.cpu().detach().numpy())})

    @torch.no_grad()
    @rank_zero_only
    def log_image(self, name: str, image: torch.Tensor, **kwargs):

        if is_rank_zero():
            image_grid = utils.make_grid(
                image, normalize=True, value_range=(-1, 1), **kwargs
            )
            wandb_image = wandb.Image(image_grid.cpu())
            self.experiment.log({name: wandb_image}, commit=False)
