from __future__ import annotations

from abc import ABC
from typing import Dict, Any, Tuple

import omegaconf
import torch.distributed

import distances
import losses
import models.prediction as prediction
from distances import compute_intersection
from models import networks
from models.general_model import GeneralModel
from models.trajectory_dict import *
from models.trajectory_dict import TrajectoryDict as TD, TrajectoryDict


class TrajectoryModel(GeneralModel, ABC):
    """
    We distinguish between "video", which is a "segment" in a trajectory in the input space, and "trajectory", which is
    a point in the latent space. They can be understood as input and output of the encoder.
    """

    def __init__(self,
                 encoder_params: omegaconf.dictconfig.DictConfig[str, Any],
                 decoder_params: omegaconf.dictconfig.DictConfig[str, Any],
                 time_encoder_strategy: str,
                 time_decoder_strategy: str,
                 optim_params: Dict[str, Any],
                 loss_params: Dict[str, Any],
                 save_results_path: str = '',
                 # In case of prediction, what experiment to run
                 predict_mode: str = None,
                 id_model=None,
                 latent_distribution='gaussian',
                 sample_mode=False,
                 distance_type='kl-divergence',
                 # Use symmetric distances, or not. If not, similarities are treated as conditional probabilities
                 symmetric_dist=True,
                 point_trajectory=False,  # Predict a point, not a distribution
                 # Number of points to sample from every predicted trajectory distribution to train decoder
                 num_sample_points=20,
                 hidden_size=256,
                 feature_size=256,
                 # In the reconstruction loss, predict points from future if input is past, or past if input is future
                 generate_extrapolation=False,
                 # Reencode decoded segments to trajectory space
                 reencode=True,
                 # How to combine the representations of the different samples once reencoded
                 option_reencode=3,
                 # Compute intersection of past and future and decode it
                 reconstruct_intersection=False,
                 # Use the whole segment (past+future) as input and in losses
                 use_all=False,
                 detach_latent=False,
                 num_classifier_layers=2,
                 classifier_params: omegaconf.dictconfig.DictConfig[str, Any] = None,
                 metrics_use=None,
                 save_to_tmp=False,
                 all_hard_positives=False,  # Explanation in losses.get_negative_pair
                 all_hard_negatives=False,
                 scale_variance: float = 1.0
                 ):

        super(TrajectoryModel, self).__init__(optim_params, loss_params, predict_mode, id_model, point_trajectory,
                                              sample_mode, latent_distribution, num_sample_points, save_to_tmp,
                                              scale_variance)

        # Create attributes
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params
        self.time_encoder_strategy = time_encoder_strategy
        self.time_decoder_strategy = time_decoder_strategy
        self.save_results_path = save_results_path
        self.distance_type = distance_type
        self.symmetric_dist = symmetric_dist
        self.num_sample_points = num_sample_points
        self.generate_extrapolation = generate_extrapolation
        self.reencode = reencode
        self.option_reencode = option_reencode
        self.reconstruct_intersection = reconstruct_intersection
        self.use_all = use_all
        self.detach_latent = detach_latent
        self.all_hard_positives = all_hard_positives
        self.all_hard_negatives = all_hard_negatives
        self.num_classifier_layers = num_classifier_layers
        assert self.num_classifier_layers >= 2, 'Classifier has at least one projection and one output layer'

        self.losses: Dict = dict(trajectory_loss=losses.TrajectoryLoss(),
                                 reconstruction_loss=losses.reconstruction_loss_traj,
                                 reconstruction_loss_mtp=losses.reconstruction_loss_mtp)

        # Create networks
        self.time_indices_encoder_embed_fn = networks.PositionalEncoder(time_encoder_strategy, hidden_size)
        if self.time_encoder_strategy == self.time_decoder_strategy:
            self.time_indices_decoder_embed_fn = self.time_indices_encoder_embed_fn
        else:
            self.time_indices_decoder_embed_fn = networks.PositionalEncoder(time_decoder_strategy, hidden_size)

        self.encoder = networks.get_network(**encoder_params, num_latent_dims=self.num_latent_params,
                                            hidden_size=hidden_size, feature_size=feature_size)
        self.use_decoder = 'reconstruction_loss' in self.loss_params.dict_losses or \
                           'reconstruction_loss_mtp' in self.loss_params.dict_losses or \
                           self.reencode
        if self.use_decoder:
            self.decoder = networks.get_network(**decoder_params, hidden_size=hidden_size, feature_size=feature_size)

        self.classifier = None
        self.diff_encoder_adv = False

        # Prepare metrics
        self.metrics_use = metrics_use
        self.metrics = {}
        try:
            distance_fn_name = self.loss_params.dict_losses.reconstruction_loss.params.distance_fn_name
        except omegaconf.errors.ConfigAttributeError:  # Distance function not defined. Only a problem if we are testing
            distance_fn_name = None
        try:
            param_dist = self.loss_params.dict_losses.reconstruction_loss.params.param_dist
        except omegaconf.errors.ConfigAttributeError:
            param_dist = None
        self.metrics_test = {
            'traj_accuracy':
                losses.TrajectoryAccuracy(latent_distribution=latent_distribution, distance_type=distance_type,
                                          num_latent_params=self.num_latent_params, return_mr=False, n=100),
            'traj_accuracy_reencoded':
                losses.TrajectoryAccuracy(latent_distribution=latent_distribution, distance_type=distance_type,
                                          num_latent_params=self.num_latent_params, return_mr=False, n=100,
                                          key='tra_dec_p_inp_p', query='tra_dec_f_inp_f'),
            'traj_accuracy_hard':
                losses.TrajectoryAccuracyHard(key='tra_inp_p', query='tra_dec_p_inp_p', negative='tra_dec_p_inp_f',
                                              latent_distribution=latent_distribution, distance_type=distance_type,
                                              num_latent_params=self.num_latent_params)
        }
        if self.metrics_use is None:  # Normal
            self.metrics_test['future_prediction'] = \
                losses.FuturePrediction(distance_fn_name=distance_fn_name, param_dist=param_dist)
        elif self.metrics_use == 'interpolation':
            self.metrics_test['interpolation_prediction'] = \
                losses.FuturePrediction(distance_fn_name=distance_fn_name, param_dist=param_dist,
                                        prediction_key='seg_dec_decode_inp_p__inter__inp_f',
                                        ground_truth_key='seg_inp_d')
        else:  # self.metrics_use == 'past'
            self.metrics_test['past_prediction'] = \
                losses.FuturePrediction(distance_fn_name=distance_fn_name, param_dist=param_dist,
                                        prediction_key='seg_dec_p_inp_f',
                                        ground_truth_key='seg_inp_p')

    def decode(self, trajectories, time_indices_decode, time_labels_decode, vid_len_decode, prior_dict, tensor_size):
        """
        We separate it from the forward so that it can be reused in some visualizations
        """
        time_embeddings_decode = self.time_indices_decoder_embed_fn(time_indices_decode)
        decoded_points = self.decoder(trajectories, time_embeddings_decode)  # [B*M, Z, Td, S, C]

        dict_decoded = TD.uncombine(decoded_points, prior_dict, tensor_size, time_steps=time_indices_decode,
                                    time_label=time_labels_decode, seg_len=vid_len_decode,
                                    last_operation='decode')
        return decoded_points, dict_decoded

    def forward(self, dict_inputs: TD, mode='train', **kwargs) -> Tuple[TrajectoryDict, Dict[Any, Any]]:
        """
        Both encoder and decoder
        """

        adv_dict = {}

        # ------------------------------- Encode videos -------------------------------- #
        video, vid_len_encode, time_indices_encode, _, tensor_size = dict_inputs.combine('encode')
        time_embeddings_encode = self.time_indices_encoder_embed_fn(time_indices_encode)  # B*M, Te, H
        encoded_trajectories = self.encoder(video, vid_len_encode, time_embeddings_encode)  # B*M, N*num_latent_params
        dict_latent = TD.uncombine(encoded_trajectories, dict_inputs, tensor_size, last_operation='encode')

        inter_name = '__inter__'.join(['inp_p', 'inp_f'])
        if self.reconstruct_intersection:
            intersection = compute_intersection(dict_latent['tra_inp_p']['tensor'], dict_latent['tra_inp_f']['tensor'],
                                                self.point_trajectory, self.num_latent_params, self.latent_distribution)
            dict_latent.add_trajectory(f'tra_{inter_name}', intersection, origin=inter_name)
            dict_latent.add_decoding_info(f'tra_{inter_name}', ['seg_inp_p', 'seg_inp_f'], dict_inputs)

        if self.generate_extrapolation:
            dict_latent.add_decoding_info('tra_inp_p', ['seg_inp_p', 'seg_inp_f'], dict_inputs)
            dict_latent.add_decoding_info('tra_inp_f', ['seg_inp_p', 'seg_inp_f'], dict_inputs)

        """
        The following generates a lot of pairs where one part of the trajectory is exactly the same and another
        is just different. In the symmetric case I have those as soft negatives, but in the asymmetric case I have to
        have them as hard negatives (P(A|B)=P(B|A)=0). May be too harsh and lead to bad training, so we do not add this 
        in the asymmetric case.
        """
        if self.use_all and self.symmetric_dist:
            dict_latent.add_decoding_info('tra_inp_a', ['seg_inp_p', 'seg_inp_f', 'seg_inp_a'], dict_inputs)
            dict_latent.add_decoding_info('tra_inp_p', ['seg_inp_a'], dict_inputs)
            dict_latent.add_decoding_info('tra_inp_f', ['seg_inp_a'], dict_inputs)

        if 'seg_inp_d' in dict_inputs:  # Decode to specific time indices
            dict_latent.add_decoding_info('tra_inp_p', ['seg_inp_d'], dict_inputs)
            dict_latent.add_decoding_info('tra_inp_f', ['seg_inp_d'], dict_inputs)
            if 'seg_inp_a' in dict_inputs:
                dict_latent.add_decoding_info('tra_inp_a', ['seg_inp_a', 'seg_inp_d'], dict_inputs)
            if self.reconstruct_intersection:
                dict_latent.add_decoding_info(f'tra_{inter_name}', ['seg_inp_d'], dict_inputs)

        all_dicts = TD.join_dicts([dict_inputs, dict_latent])

        if self.use_decoder:
            # ---------------------------- Decode trajectories ----------------------------- #
            if self.detach_latent:
                dict_latent = dict_latent.detach()

            sample_fn = self.sample_distribution if mode in ['', 'test'] else None
            encoded_trajectories, vid_len_decode, time_indices_decode, time_labels_decode, tensor_size = \
                dict_latent.combine(purpose='decode', reference_dict=dict_inputs, sample_fn=sample_fn)
            if mode in ['', 'test']:
                sampled_trajs = encoded_trajectories
            else:
                sampled_trajs = self.sample_distribution(encoded_trajectories)  # [B*M, N, self.num_sample_points]
            decoded_points, dict_decoded = self.decode(sampled_trajs, time_indices_decode, time_labels_decode,
                                                       vid_len_decode, dict_latent, tensor_size)
            all_dicts = TD.join_dicts([all_dicts, dict_decoded])

            if self.reencode:
                # ---------------------------- Re-encode trajectories ----------------------------- #
                # Adapt to format of encoder.
                # M is the number of trajectories per batch element
                time_embeddings_reencode = self.time_indices_encoder_embed_fn(time_indices_decode)
                BM, T, S, D = decoded_points.shape
                H = time_embeddings_reencode.shape[-1]
                time_embeddings_reencode_ = time_embeddings_reencode.view(-1, *time_embeddings_reencode.shape[2:])
                # time_embeddings_reencode_ : [BxMxT, H]
                vid_len_reencode = vid_len_decode

                """With respect to the different sampled points, we have three options. 1) only use the first sample 
                from all self.num_sample_points. It would be like having self.num_sample_points=1. An alternative would 
                be to sample a different one (randomly) for every element in the batch. 2) Compute encoding for all of 
                them, and use in the loss. 3) Compute encoding for all of them, but average them before using in the 
                loss."""
                if self.option_reencode == 1:
                    decoded_points_ = decoded_points[..., 0, :]  # [BxM, T, D]
                    time_embeddings_reencode_ = time_embeddings_reencode_.reshape(-1, T, H)  # [BxM, T, H]
                else:  # options 2 and 3
                    decoded_points_ = decoded_points.permute(0, 2, 1, 3).reshape(-1, T, D)  # [BxMxS, T, D]
                    time_embeddings_reencode_ = time_embeddings_reencode_.unsqueeze(1).repeat(1, S, 1)  # [BxZxT, S, H]
                    time_embeddings_reencode_ = time_embeddings_reencode_.view(-1, T, S, H)  # [BxM, T, S, H]
                    time_embeddings_reencode_ = time_embeddings_reencode_.permute(0, 2, 1, 3)  # [BxM, S, T, H]
                    time_embeddings_reencode_ = time_embeddings_reencode_.reshape(-1, T, H)  # [BxMxS, T, H]
                    vid_len_reencode = vid_len_reencode.unsqueeze(-1).repeat(1, S).reshape(-1)  # [BxMxS]

                reencoded_points = self.encoder(decoded_points_, vid_len_reencode, time_embeddings_reencode_)
                if self.option_reencode == 1:
                    reencoded_points = reencoded_points.reshape(BM, reencoded_points.shape[-1])
                else:
                    reencoded_points = reencoded_points.reshape(BM, S, reencoded_points.shape[-1])
                    if self.option_reencode == 3:
                        reencoded_points = distances.compute_average(reencoded_points, self.latent_distribution,
                                                                     self.num_latent_params, dim=-2)

                dict_reencoded = TD.uncombine(reencoded_points, dict_decoded, tensor_size, last_operation='encode')

                all_dicts = TD.join_dicts([all_dicts, dict_reencoded])

        return all_dicts, adv_dict

    def step(self, batch: Dict[str, torch.tensor], batch_idx: int, mode: str = 'train', only_forward: bool = False):

        dict_inputs = TrajectoryDict.create_from_batch(batch, return_all=self.use_all)
        all_dicts, adv_dict = self.forward(dict_inputs, mode=mode)

        if only_forward:
            return all_dicts

        # ---------------------------Compute losses --------------------------- #
        loss_dict = {}
        loss = torch.zeros(1).to(batch['index'].device)
        for_metrics = {'all_dicts': all_dicts}
        for loss_name, loss_params in self.loss_params.dict_losses.items():
            if loss_params['λ'] == 0:
                continue
            kwargs = {} if loss_params['params'] is None else dict(loss_params['params'])
            kwargs['num_latent_params'] = self.num_latent_params
            kwargs['latent_distribution'] = self.latent_distribution
            loss_dict[loss_name], to_report, for_metrics_ = \
                self.losses[loss_name](**all_dicts,
                                       **adv_dict,
                                       distance_type=self.distance_type,
                                       generate_extrapolation=self.generate_extrapolation,
                                       reencode=self.reencode,
                                       reconstruct_intersection=self.reconstruct_intersection,
                                       use_all=self.use_all,
                                       symmetric_dist=self.symmetric_dist,
                                       **kwargs)
            loss += loss_params['λ'] * loss_dict[loss_name]

            self.log(f'{mode}/loss_{loss_name}', loss_dict[loss_name].item(), prog_bar=True)
            self.report(to_report, mode)
            for_metrics = {**for_metrics, **for_metrics_}  # In python 3.9 this will be  for_metrics | for_metrics_

            if self.reconstruct_intersection and self.reencode:
                self.log(f'{mode}/mean_intersection',
                         all_dicts['tra_dec_p_inp_p__inter__inp_f']['tensor'].pow(2).sum(-1).sqrt().mean().detach(),
                         prog_bar=True)

        self.log(f'{mode}/loss', loss.item(), prog_bar=True)
        self.compute_metrics(for_metrics, mode)

        # Optionally, log other values, like magnitudes of the outputs
        # self.logger.log_hist(f'{mode}/magnitude_output', outputs['...'].pow(2).sum(-1).sqrt())

        # Always return loss, and optionally other information
        return {'loss': loss, 'for_metrics': for_metrics}

    def training_step(self, batch: Dict[str, torch.tensor], batch_idx: int):
        loss_dict = self.step(batch, batch_idx, 'train')
        return loss_dict

    def predict_step(self, batch: Dict[str, torch.tensor], batch_idx: int, **kwargs):
        # When calling any of these, make sure the dataset split being used is the desired one (typically either 'test'
        # or 'all'). Default is 'all'. To change it, in the config specify "dataset.split_use: test"

        if self.predict_mode == 'visualize_gradcam':
            prediction.gradcam(self, batch, self.latent_distribution)
        else:
            mode = 'visualize_pca' if 'visualize_pca' in self.predict_mode else ''
            all_dicts = self.step(batch, batch_idx, mode=mode, only_forward=True)

            if self.predict_mode in ['visualize_trajectories']:
                to_save = [batch['index'].cpu(),
                           batch['time_indices_past'].cpu(),
                           batch['time_indices_future'].cpu(),
                           batch['video_len_past'].cpu(),
                           batch['video_len_future'].cpu(),
                           all_dicts['seg_inp_p']['tensor'].cpu(),
                           all_dicts['seg_inp_f']['tensor'].cpu(),
                           all_dicts['tra_inp_p']['tensor'].cpu(),
                           all_dicts['tra_inp_f']['tensor'].cpu(),
                           all_dicts['seg_dec_p_inp_p']['tensor'].cpu(),
                           all_dicts['seg_dec_f_inp_p']['tensor'].cpu(),
                           all_dicts['seg_dec_p_inp_f']['tensor'].cpu(),
                           all_dicts['seg_dec_f_inp_f']['tensor'].cpu()]
                save_names = ['sample_ids', 'time_indices_past', 'time_indices_future', 'video_len_past',
                              'video_len_future', 'seg_inp_p', 'seg_inp_f', 'tra_inp_p', 'tra_inp_f', 'seg_dec_p_inp_p',
                              'seg_dec_f_inp_p', 'seg_dec_p_inp_f', 'seg_dec_f_inp_f']
                if self.reconstruct_intersection:
                    to_save.append(all_dicts['seg_dec_p_inp_p__inter__inp_f']['tensor'].cpu())
                    to_save.append(all_dicts['seg_dec_f_inp_p__inter__inp_f']['tensor'].cpu())
                    save_names += ['seg_dec_p_inp_p__inter__inp_f', 'seg_dec_f_inp_p__inter__inp_f']
                if 'time_indices_past_decode' in batch:
                    to_save.append(batch['time_indices_past_decode'].cpu())
                    to_save.append(batch['video_len_past_decode'].cpu())
                    to_save.append(batch['time_indices_future_decode'].cpu())
                    to_save.append(batch['video_len_future_decode'].cpu())
                    save_names += ['time_indices_past_decode', 'video_len_past_decode', 'time_indices_future_decode',
                                   'video_len_future_decode']
                if 'seg_inp_d' in all_dicts:  # Specific decoding times
                    for k in [k for k in all_dicts.keys() if k.startswith('seg_dec_decode')]:
                        to_save.append(all_dicts[k]['tensor'].cpu())
                        save_names.append(k)
                    to_save += [batch['time_indices_decode'].cpu(), batch['video_len_decode'].cpu()]
                    save_names += ['time_indices_decode', 'video_len_decode']
            elif self.predict_mode in ['visualize_intersection']:
                save_names = ['sample_ids', 'tra_inp_p', 'tra_inp_f']
                to_save = [batch['index'].cpu(),
                           all_dicts['tra_inp_p']['tensor'].cpu(),
                           all_dicts['tra_inp_f']['tensor'].cpu()]
            elif self.predict_mode.startswith('visualize_pca_increment'):
                if 'tra_inp_a' in all_dicts:  # Reconstruct from 'all'
                    save_names = ['sample_ids', 'tra_inp_a', 'seg_inp_a', 'time_indices_all']
                    to_save = [batch['index'].cpu(), all_dicts['tra_inp_a']['tensor'].cpu(),
                               all_dicts['seg_inp_a']['tensor'].cpu(), batch['time_indices_all'].cpu()]
                elif 'tra_inp_p__inter__inp_f' in all_dicts:  # Reconstruct from intersection
                    save_names = ['sample_ids', 'tra_inp_p__inter__inp_f', 'seg_inp_p', 'time_indices_all']
                    to_save = [batch['index'].cpu(), all_dicts['tra_inp_p__inter__inp_f']['tensor'].cpu(),
                               all_dicts['seg_inp_p']['tensor'].cpu(), batch['time_indices_past'].cpu()]
                else:
                    raise KeyError('Either the intersection of "all" have to be computed')

            else:
                raise KeyError

            self.save_tensors(to_save)
            self.save_tensors.tensor_names = save_names

    def on_predict_epoch_end(self, outputs) -> None:
        if 'visualize' in self.predict_mode:
            prediction.visualize(self.trainer.datamodule.predict_dataloader().dataset, self.predict_mode,
                                 self.save_tensors, self.reconstruct_intersection, self.sample_distribution,
                                 self.distance_type, self.num_latent_params, self.point_trajectory, self.id_model,
                                 self.latent_distribution, decode_fn=self.decode)

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
