from __future__ import annotations

from abc import ABC
from typing import Dict, Any

import omegaconf
import torch.distributed

import losses
from models import networks
from models import prediction
from models.general_model import GeneralModel
from models.networks.baselines_networks import TrajEncoder
from models.trajectory_dict import *


class VRNNModel(GeneralModel, ABC):
    def __init__(self,
                 encoder_params: omegaconf.dictconfig.DictConfig[str, Any],
                 decoder_params: omegaconf.dictconfig.DictConfig[str, Any],
                 optim_params: Dict[str, Any],
                 loss_params: Dict[str, Any],
                 point_trajectory=False,
                 sample_mode=False,
                 latent_distribution='gaussian',
                 num_sample_points=1,
                 save_results_path: str = '',
                 # In case of prediction, what experiment to run
                 predict_mode: str = None,
                 id_model=None,
                 hidden_size=256,
                 feature_size=256,
                 trajectron=False,
                 metrics_use=None,
                 save_to_tmp=False
                 ):

        super(VRNNModel, self).__init__(optim_params, loss_params, predict_mode, id_model, point_trajectory,
                                        sample_mode, latent_distribution, num_sample_points, save_to_tmp)

        # if not sample_mode:
        #     assert self.num_sample_points == 1, 'The VRNN and Trajectron work with only one sample'

        # Create attributes
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params
        self.save_results_path = save_results_path
        self.hidden_size = hidden_size
        self.trajectron = trajectron
        self.metrics_use = metrics_use

        self.losses: Dict = dict(kld_loss=losses.kld_loss,
                                 reconstruction_loss=losses.reconstruction_loss_categorical,
                                 info_loss=losses.info_loss)

        # Create networks
        self.x_encoder = TrajEncoder(encoder_params['in_size'], hidden_size)
        self.encoder = networks.get_network(**encoder_params, hidden_size=hidden_size, feature_size=feature_size,
                                            num_latent_params=self.num_latent_params,
                                            sample_fn=self.sample_distribution)
        input_size_dec = 0 if not trajectron else feature_size
        self.decoder = networks.get_network(**decoder_params, in_size=input_size_dec, hidden_size=hidden_size,
                                            feature_size=feature_size)

        try:
            distance_fn_name = self.loss_params.dict_losses.reconstruction_loss.params.distance_fn_name
        except omegaconf.errors.ConfigAttributeError:  # Distance function not defined. Only a problem if we are testing
            distance_fn_name = None
        self.metrics = {}
        try:
            param_dist = self.loss_params.dict_losses.reconstruction_loss.params.param_dist
        except omegaconf.errors.ConfigAttributeError:
            param_dist = None
        self.metrics_test = {'future_prediction': losses.FuturePrediction(distance_fn_name=distance_fn_name,
                                                                          param_dist=param_dist)}

    @staticmethod
    def mask_duration(duration, length, multiple_steps, duration_past):
        """
        Create a mask to mask out the temporal padding.
        Some t_k should not be predicted because they are outside the range of "duration".
        If multiple_steps, The predictions and GT have a shape like [t1, t2, t3, ..., tN, t2, t3, ..., tN, t3, ...]

        If duration_past is not None, we only return the prediction that we would have gotten from having input the
        past trajectory
        """
        aux = torch.arange(1, length)
        if multiple_steps:
            aux = aux[None, None, :].expand(duration.shape[0], length - 1, length - 1)
            if duration_past is not None:
                # At inference time we only predict the sequence that starts when the past finishes (Trajectron++)
                not_use = torch.ones_like(aux)
                not_use[torch.arange(duration_past.shape[0]), duration_past - 1] = 0
                aux = aux.clone()  # Otherwise, following inplace operation does not work
                aux[torch.where(not_use.bool())] = length  # just a large number
            indices = torch.triu_indices(length - 1, length - 1, 0)
            aux = aux[:, indices[0], indices[1]]
        else:
            aux = aux[None, :].expand(duration.shape[0], length - 1)
            if duration_past is not None:
                # At inference time we don't want to predict the past. At training time yes because it is AR (VRNN)
                indices_past = aux < duration_past[:, None].cpu()
                aux = aux.clone()
                aux[indices_past] = length  # just a large number
            aux = aux.to(duration.device)

        indices_fill = None
        if duration_past is not None:
            aux_2 = torch.arange(0, length)
            # if multiple_steps:
            aux_2 = aux_2[None, :].expand(duration.shape[0], length).to(duration.device)
            indices_fill = aux_2 < (duration - duration_past)[:, None]

        mask = aux.to(duration.device) < duration[:, None]
        if indices_fill is not None:
            assert (mask.sum(-1) == indices_fill.sum(-1)).all()
        return mask, indices_fill

    def forward(self, x, duration, duration_past=None):

        z_distr, prior_distr, y = [], [], []

        x_feat = self.x_encoder(x)

        h = None
        length = x.shape[1]
        for t in range(length):

            z_distr_t, z_feat_t, prior_distr_t, h = self.encoder(x_feat[:, t], h, training=self.training,
                                                                 first_step=t == 0)
            features_context = None
            if self.trajectron:
                # "points" are not really temporal points, just additional information, in this case past context
                features_context = h[-1].unsqueeze(-2).expand(h[-1].shape[0], z_feat_t.shape[-2], h.shape[-1])
            y_t = self.decoder(z_feat_t, features_context, gt_x=x_feat[:, t + 1:])

            if len(z_feat_t.shape) > len(z_distr_t.shape) and not self.trajectron:
                z_distr_t = z_distr_t[:, None].expand(*z_feat_t.shape[:2], z_distr_t.shape[-1])
                prior_distr_t = prior_distr_t[:, None].expand(*z_feat_t.shape[:2], prior_distr_t.shape[-1])
            z_distr.append(z_distr_t)
            prior_distr.append(prior_distr_t)
            y.append(y_t)

        z_distr = torch.stack(z_distr, dim=1)
        prior_distr = torch.stack(prior_distr, dim=1)

        multiple_steps = type(y[0]) == list
        if multiple_steps:  # We make more than one prediction (step into the future) at every time-step
            y_all = []
            x_all = []
            z_distr_all = []
            for t in range(len(y)):
                for t_future in range(len(y[t])):
                    x_all.append(x[:, t + t_future + 1])  # ground truth
                    y_all.append(y[t][t_future])
                    z_distr_all.append(z_distr[:, t])
            x = torch.stack(x_all, dim=1)
            y = torch.stack(y_all, dim=1)
            z_distr_reconstruct = torch.stack(z_distr_all, dim=1)
        else:
            y = torch.stack(y, dim=1)  # [B, T, H]
            # The target is the next step. The last y does not have a ground truth target
            # We should not even input the last x to the rnn during training. But it's fine.
            y = y[:, :-1]
            x = x[:, 1:]
            z_distr_reconstruct = z_distr

        mask, indices_fill = self.mask_duration(duration, length, multiple_steps, duration_past)
        if indices_fill is not None:  # Only used for prediction purposes
            pred = torch.zeros([x.shape[0], length, self.num_sample_points, x.shape[-1]]).to(x.device)
            pred[indices_fill] = y[mask].type(pred.type())
            y = pred
            target = None
        else:
            target = x[mask]  # [B', H]. B' encompasses T
            y = y[mask]  # [B', (feature_size),  H]
            if multiple_steps:
                z_distr_reconstruct = z_distr_reconstruct[mask]

        # z_distr and z_distr_reconstruct are the same, just different format

        return z_distr, prior_distr, y, target, z_distr_reconstruct

    def step(self, batch: Dict[str, torch.tensor], batch_idx: int, mode: str = 'train', only_forward: bool = False):

        if mode == '' or mode == 'test':  # prediction
            x = batch['past']  # This will be used as input
            target = batch['interpolation'] if self.metrics_use == 'interpolation' else batch['future']
            duration_past = batch['video_len_past']
            # Duration of the future, for accuracy metrics
            duration_pred = batch['video_len_decode'] if self.metrics_use == 'interpolation' else \
                batch['video_len_future']
        else:
            # More convenient to use 'all'. 'all' length will always be  <= len(past) + len(future)
            x = batch['all']
            duration_past = target = None
            duration_pred = batch['video_len_future']

        duration = batch['video_len_past'] + duration_pred
        z_distr, prior_distr, y, target_, z_distr_reconstruct = self.forward(x, duration, duration_past)
        if target_ is not None:
            target = target_

        if only_forward:
            return y, batch['index']

        # ---------------------------Compute losses --------------------------- #
        loss_dict = {}
        loss = torch.zeros(1).to(batch['index'].device)
        for_metrics = {'query': y, 'target': target, 'duration': duration_pred}
        for loss_name, loss_params in self.loss_params.dict_losses.items():
            if loss_params['λ'] == 0:
                continue
            kwargs = {} if loss_params['params'] is None else dict(loss_params['params'])

            loss_dict[loss_name], to_report, for_metrics_ = \
                self.losses[loss_name](query=y, target=target, z_distr=z_distr, z_distr_reconstruct=z_distr_reconstruct,
                                       prior_distr=prior_distr, latent_distribution=self.latent_distribution,
                                       num_latent_params=self.num_latent_params, **kwargs)
            loss += loss_params['λ'] * loss_dict[loss_name]

            self.log(f'{mode}/loss_{loss_name}', loss_dict[loss_name].item(), prog_bar=True)
            self.report(to_report, mode)
            for_metrics = {**for_metrics, **for_metrics_}  # In python 3.9 this will be  for_metrics | for_metrics_

        self.log(f'{mode}/loss', loss.item(), prog_bar=True)
        self.compute_metrics(for_metrics, mode)

        return {'loss': loss, 'for_metrics': for_metrics}

    def training_step(self, batch: Dict[str, torch.tensor], batch_idx: int):
        return self.step(batch, batch_idx, 'train')

    def predict_step(self, batch: Dict[str, torch.tensor], batch_idx: int, **kwargs):
        y, _ = self.step(batch, batch_idx, mode='', only_forward=True)

        if self.predict_mode in ['visualize_trajectories']:
            to_save = [batch['index'].cpu(),
                       batch['time_indices_past'].cpu(),
                       batch['time_indices_future'].cpu(),
                       batch['video_len_past'].cpu(),
                       batch['video_len_future'].cpu(),
                       batch['past'].cpu(),
                       batch['future'].cpu(),
                       y.cpu()]
            save_names = ['sample_ids', 'time_indices_past', 'time_indices_future', 'video_len_past', 'y']
        else:
            raise KeyError

        self.save_tensors(to_save)
        self.save_tensors.tensor_names = save_names

    def on_predict_epoch_end(self, outputs) -> None:
        if 'visualize' in self.predict_mode:
            prediction.visualize(self.trainer.datamodule.predict_dataloader().dataset, self.predict_mode,
                                 self.save_tensors, reconstruct_intersection=False, model_id=self.id_model,
                                 vrnn_model=True)

    def test_step_end(self, outputs):
        """
        Unlike test_step, this combines different gpus
        """
        # update and log
        for_metrics = outputs['for_metrics']
        for name_metric, metric in self.metrics_test.items():
            val = metric(**for_metrics)
            if val is not None:
                self.log(name_metric, val, sync_dist=True)
