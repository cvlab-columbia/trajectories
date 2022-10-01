"""
The separation between GeneralModel and MyModel(s) depends on the specific project. Some methods will be more
conveniently defined in one class or the other depending on what is the level of variation across MyModels(s)
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Union, List, Dict, Any

import omegaconf
import torch.distributed
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule

import distances
import losses
from models.trajectory_dict import *
from utils.optimizer import construct_optimizer


@dataclass(eq=False)  # Adds automatically generated methods like __init__. Do not create an eq() method
class GeneralModel(LightningModule, abc.ABC):
    """
    Generic LightningModule for the project. Do not instantiate
    """
    optim_params: omegaconf.dictconfig.DictConfig[str, Any]
    loss_params: omegaconf.dictconfig.DictConfig[str, Any]
    predict_mode: str = None
    id_model: str = None
    point_trajectory: bool = False
    sample_mode: bool = False  # Sample the mode of the distribution. For inference purposes
    latent_distribution: str = 'gaussian'
    num_sample_points: int = 1
    save_to_tmp: bool = False  # Save test results to tmp folder
    scale_variance: float = 1.0  # Scale the variance when sampling. Generates more diverse samples if > 1.0

    def __post_init__(self):
        super().__init__()
        self.save_hyperparameters()  # To load from checkpoint when resuming
        self.losses: Dict = dict()
        self.metrics: Dict = dict()
        self.metrics_test: Dict = dict()
        if self.predict_mode is not None:
            self.save_tensors = losses.SaveTensors()

        # Figure out the number of latent params
        if self.point_trajectory or self.latent_distribution == 'categorical':
            num_latent_params = 1
        elif self.latent_distribution in ['gaussian', 'bbox']:
            num_latent_params = 2  # mean and variance. Not modeling covariance
        else:
            raise NotImplementedError
        self.num_latent_params = num_latent_params

    @abc.abstractmethod
    def forward(self, **kwargs):
        pass

    @abc.abstractmethod
    def step(self, batch: Dict[str, torch.tensor], batch_idx: int, mode: str = 'train', only_forward: bool = False) \
            -> Dict:
        pass

    def training_step(self, batch: Dict[str, torch.tensor], batch_idx: int):
        loss_dict = self.step(batch, batch_idx, 'train')
        return loss_dict

    def validation_step(self, batch: Dict[str, torch.tensor], batch_idx: int):
        return self.step(batch, batch_idx, 'validate')

    def test_step(self, batch: Dict[str, torch.tensor], batch_idx: int):
        return self.step(batch, batch_idx, 'test')

    def configure_optimizers(self) -> Union[optim, List[optim]]:
        return construct_optimizer(self, self.optim_params)

    def get_progress_bar_dict(self) -> Dict[str, Union[int, str]]:
        # call .item() only once but store elements without graphs
        running_train_loss = self.trainer.fit_loop.running_loss.mean()
        avg_training_loss = None
        if running_train_loss is not None:
            avg_training_loss = running_train_loss.cpu().item()
        elif self.automatic_optimization:
            avg_training_loss = float("NaN")

        tqdm_dict = {}
        if avg_training_loss is not None:
            tqdm_dict["loss"] = f"{avg_training_loss:.3g}"

        # Print times from profiler
        if "get_train_batch" in self.trainer.profiler.recorded_durations:
            data_time = self.trainer.profiler.recorded_durations["get_train_batch"][-1]
            data_time_avg = np.mean(self.trainer.profiler.recorded_durations["get_train_batch"])
            tqdm_dict["data_time"] = f"{data_time:.3g} ({data_time_avg:.3g})"
        if "run_training_batch" in self.trainer.profiler.recorded_durations:
            batch_time = self.trainer.profiler.recorded_durations["run_training_batch"][-1]
            batch_time_avg = np.mean(self.trainer.profiler.recorded_durations["run_training_batch"])
            tqdm_dict["total_batch_time"] = f"{batch_time:.3g} ({batch_time_avg:.3g})"
        return tqdm_dict

    def on_test_epoch_end(self):
        all_results = {}
        all_metrics = {}
        for name_metric, metric in self.metrics_test.items():
            val = metric.compute()
            self.log(name_metric, val, sync_dist=True)
            all_metrics[name_metric] = val
            print(name_metric, val)
            for k, v in val.items():
                all_results[name_metric + '_' + k] = v
        if self.save_to_tmp:
            torch.save(all_metrics, '/tmp/tmp_results.pth')

    def sample_distribution(self, dist_trajs, grad_through=True, return_all=False, sample_mode=None):
        """
        Sample from a distribution
        :param sample_mode: overwrites self.sample_mode
        :param dist_trajs: [B, N*num_latent_params]
        :param grad_through: Requires gradient to flow through operation. If False, gradient can still flow. This flag
            is a requirement to force it to flow.
        :param return_all: returns all possible samples. It only applies to discrete distributions (categorical).
            Overwritten by sample_mode == True
        :return: sampled_trajs [B, self.num_sample_points, N]
        """
        if sample_mode is None:
            sample_mode = self.sample_mode

        if self.point_trajectory:
            assert self.num_sample_points == 1, 'It is not a distribution, more than one point will not be sampled'
            return dist_trajs.unsqueeze(1)

        param_1, param_2 = distances.get_params(dist_trajs, self.num_latent_params)
        if self.latent_distribution == 'gaussian':
            mean, logvar = param_1, param_2
            if sample_mode:
                # assert self.num_sample_points == 1
                return mean.unsqueeze(1)
            std = (logvar / 2).exp()
            dims = [*mean.shape[:-1], self.num_sample_points, mean.shape[-1]]
            sampled_trajs = std[..., None, :] * torch.randn(dims, device=dist_trajs.device) + mean[..., None, :]

        elif self.latent_distribution == 'bbox':
            """
            Uniform sampling, independent for every dimension
            We treat the parameters as the minimum (z) and maximum (Z) of the box (NOT center + delta)
            Note that when Z < z (for any dimension), we have a "negative" box, but the sampling is done anyway between 
            the values of z and Z. It is the responsibility of other parts of the code to prevent that from happening
            """
            min_box, max_box = param_1, param_2
            if sample_mode:
                assert self.num_sample_points == 1
                return ((min_box + max_box) / 2).unsqueeze(1)
            dims = [*min_box.shape[:-1], self.num_sample_points, min_box.shape[-1]]
            random_uniform = torch.rand(dims, device=dist_trajs.device) * self.scale_variance
            sampled_trajs = min_box[..., None, :] + random_uniform * (max_box - min_box)[..., None, :]

        elif self.latent_distribution == 'categorical':
            p = losses.categorical_softmax(param_1, dim=-1)
            if sample_mode:
                assert not grad_through, 'Can only sample mode in categorical variables in an inference setting'
                if self.num_sample_points == 1:
                    sampled_trajs = p.argmax(-1).unsqueeze(-1)
                else:  # Also works for 1 sample point, but the previous line makes it more clear
                    sampled_trajs = p.topk(self.num_sample_points)[1]
                sampled_trajs = F.one_hot(sampled_trajs, num_classes=p.shape[-1]).to(p)
            elif return_all:
                assert self.num_sample_points == 1
                sampled_trajs = torch.eye(dist_trajs.shape[-1]).to(p)
                sampled_trajs = sampled_trajs.unsqueeze(0).unsqueeze(0)
                sampled_trajs = sampled_trajs.expand(dist_trajs.shape[0], 1, dist_trajs.shape[-1], dist_trajs.shape[-1])
            elif grad_through:
                p = p.unsqueeze(-2).expand(*p.shape[:-1], self.num_sample_points, p.shape[-1])
                sampled_trajs = F.gumbel_softmax(p, tau=1, hard=True)
            else:
                sampled_trajs = torch.multinomial(p, self.num_sample_points, True)
                sampled_trajs = F.one_hot(sampled_trajs, num_classes=p.shape[-1]).to(p)

        else:
            raise NotImplementedError

        return sampled_trajs

    def report(self, to_report, mode):
        if len(to_report) > 0:
            for name_report, (type_report, value_report) in to_report.items():
                if type_report == 'histogram':
                    try:
                        self.logger.log_hist(f'{mode}/{name_report}', value_report.detach())
                    except Exception as e:
                        print(f'There are some bad values in the histogram ({e})')
                else:
                    self.log(f'{mode}/{name_report}', value_report.item(), prog_bar=True)

    def compute_metrics(self, for_metrics, mode):
        for_metrics = {k: v.detach() if type(v) == torch.Tensor else v for k, v in for_metrics.items()}
        if mode != 'test':
            # If test, we compute this in test_step_end
            metrics = self.metrics
            for metric_name, metric_fn in metrics.items():
                results = metric_fn(**for_metrics)
                for k, v in results.items():
                    if type(v) in [int, float] or torch.tensor(v.shape).prod() == 1:
                        self.log(f'{mode}/{metric_name}_{k}', float(v), prog_bar=True)
