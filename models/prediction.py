"""
Methods to be called in the prediction phase. All of these are called from the LightningModule class, but we put them
in a separate file to make the different files easier to read.
These methods have two components: the step (for every batch), and the on_predict_end (join all batches)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

import distances
from models.trajectory_dict import TrajectoryDict


def visualize(dataset, predict_mode, save_tensors, reconstruct_intersection=False, sample_distribution_fn=None,
              distance_type=None, num_latent_params=2, point_trajectory=False, model_id=None, latent_distribution=None,
              vrnn_model=False, decode_fn=None):
    if predict_mode in ['visualize_trajectories']:
        saved_tensors = save_tensors.compute()
        save_names = save_tensors.tensor_names
        dataset.visualize_trajectories(saved_tensors, save_names, reconstruct_intersection, model_id, vrnn_model)
    elif predict_mode == 'visualize_intersection':
        visualize_intersection_end(save_tensors, sample_distribution_fn, num_latent_params, point_trajectory,
                                   latent_distribution)
    elif predict_mode.startswith('visualize_pca_increment'):
        visualize_pca_increment_end(save_tensors, num_latent_params, latent_distribution, distance_type, decode_fn,
                                    dataset, model_id, predict_mode)


def visualize_intersection_end(save_tensors, sample_distribution_fn, num_latent_params, point_trajectory,
                               latent_distribution, plot_viz=True):
    prob_fn = distances.get_prob_fn(latent_distribution)

    sample_ids, tra_inp_p, tra_inp_f = save_tensors.compute()
    past = {'tensor': tra_inp_p.float(), 'origin': 'past'}
    future = {'tensor': tra_inp_f.float(), 'origin': 'future'}
    intersection = distances.compute_intersection(past['tensor'], future['tensor'], point_trajectory,
                                                  num_latent_params, latent_distribution)
    samples = sample_distribution_fn(tra_inp_p)
    param_1_past, param_2_past = distances.get_params(tra_inp_p, num_latent_params)
    param_1_future, param_2_future = distances.get_params(tra_inp_f, num_latent_params)
    param_1_int, param_2_int = distances.get_params(intersection, num_latent_params)
    log_prob = prob_fn(param_1_past.float()[:, None], param_2_past.float()[:, None], samples)
    log_prob_intersection = prob_fn(param_1_int.float()[:, None], param_2_int.float()[:, None], samples)

    print(log_prob.mean(), log_prob_intersection.mean())

    if plot_viz:
        num_samples_viz = 2
        fig, axes = plt.subplots(param_1_past.shape[-1], num_samples_viz, dpi=100,
                                 figsize=(num_samples_viz * 3, param_1_past.shape[-1]))
        var_past = (param_2_past.float() / 2).exp()
        for i in range(num_samples_viz):
            # Plot the first sample, all dimensions separately
            lower_bound = param_1_past[i].min() - 1 * var_past[i].max()
            upper_bound = param_1_past[i].max() + 1 * var_past[i].max()
            resolution = 500
            x = torch.linspace(lower_bound, upper_bound, resolution)
            p_1_past_, p_2_past_ = param_1_past.float()[i][:, None, None], param_2_past.float()[i][:, None, None]
            p_1_fut_, p_2_fut_ = param_1_future.float()[i][:, None, None], param_2_future.float()[i][:, None, None]
            p_1_int_, p_2_int_ = param_1_int.float()[i][:, None, None], param_2_int.float()[i][:, None, None]
            dist_past = prob_fn(p_1_past_, p_2_past_, x[None, :, None])
            dist_future = prob_fn(p_1_fut_, p_2_fut_, x[None, :, None])
            dist_int = prob_fn(p_1_int_, p_2_int_, x[None, :, None])
            for j in range(dist_past.shape[0]):  # for all dimensions
                axes[j][i].plot(x, dist_past[j].exp(), c='g')
                axes[j][i].plot(x, dist_future[j].exp(), c='r')
                axes[j][i].plot(x, dist_int[j].exp(), c='b')

        plt.show()


def visualize_pca_increment_end(save_tensors, num_latent_params, latent_distribution, distance_type, decode_fn,
                                dataset, model_id, predict_mode):
    """
    Computes the principal components for every GROUP of trajectories. Each group consists of a base trajectory modified
    according to some attribute (there can be an increment in speed, position, etc.). This function return the most
    important dimension for that attribute (the dimensions that change the most in that attribute)
    """
    sample_ids, tra_inp_a, seg_inp_a, time_indices = save_tensors.compute()
    num_per_sample = sample_ids[:, 1].max() + 1
    tra_inp_a = tra_inp_a.view(-1, num_per_sample, tra_inp_a.shape[-1])
    param_1, param_2 = distances.get_params(tra_inp_a, num_latent_params)

    k = []
    for i in range(tra_inp_a.shape[0]):
        pca = PCA(n_components=1)
        pca.fit(param_1[i])
        k.append(pca.components_[0])

    *_, opt, delta = predict_mode.split('_')
    delta = float(delta)

    if opt == 'speed':
        index_original = int(np.round((num_per_sample - 1) / 3))  # The index where the original trajectory is
    else:  # offset
        index_original = 0

    k = np.stack(k)
    k = k.mean(0)  # k is the direction that changes the parameter we are modifying.
    important_dims = np.where(k > k.max() / 2)[0]

    alpha = torch.arange(dataset.num_copies)[None, :, None]
    # We take the original sequence and modify it gradually
    trajectories = (param_1[:, index_original:index_original + 1] + alpha * k * delta).cuda().float()
    num_steps = time_indices.shape[-1] * 2
    time_indices_decode = torch.arange(num_steps)[None, :].expand(param_1.shape[0], num_steps).cuda().float()
    time_labels_decode = ['custom']
    vid_len_decode = torch.tensor([num_steps]).expand(param_1.shape[0])
    tensor_size = torch.tensor([param_1.shape[0]])  # We treat the num_steps as different samples

    # First, only take one copy out of the num_per_sample we have (this num_per_sample is defined in collate_fn)
    ground_truth = seg_inp_a.view(-1, num_per_sample, *seg_inp_a.shape[1:])[:, index_original]
    time_indices = time_indices.view(time_indices_decode.shape[0], -1, *time_indices.shape[1:])[:, index_original]

    prior_dict = TrajectoryDict({'custom': {'tensor': None, 'space': 'traj', 'time_label': None, 'time_steps': None,
                                            'seg_len': None, 'origin': 'custom', 'splitting': 1}})
    decoded_points, dict_decoded = decode_fn(trajectories, time_indices_decode, time_labels_decode, vid_len_decode,
                                             prior_dict, tensor_size)

    time_indices_decode = time_indices_decode.cpu()

    # Adapt ground truth times to our decoding times
    aux_1 = time_indices_decode[..., None] <= time_indices[:, None, :]
    aux_2 = aux_1.float().argmax(dim=-1)  # Argmax returns the first index
    ground_truth = ground_truth[torch.arange(ground_truth.shape[0]).unsqueeze(1), aux_2]

    times = ('all', time_indices_decode, torch.tensor([num_steps] * time_indices_decode.shape[0]))
    if dataset.clean_viz:
        options = [('prediction', decoded_points)]
    else:
        options = [('ground truth', ground_truth), ('prediction', decoded_points)]

    print(f'Most important dimensions: {important_dims}')

    sample_ids = sample_ids.view(-1, num_per_sample, 2)[:, 0, 0]
    for i in range(100):
        trajectories = {}
        for option in options:
            name, tensor = option
            traj = tensor[i, :times[2][i]]
            traj = traj.view(*traj.shape[:-1], 25, 2).cpu()
            trajectories[name] = traj

        timestamps = times[1][i][:times[2][i]]
        timestamps = timestamps - timestamps[0]  # First timestamp start at 0
        timestamps = timestamps / 25  # fps = 25
        timestamps = (timestamps, ('all',) * num_steps)

        # Normalize so that it occupies all axis
        x, y = trajectories['prediction'][..., 0], trajectories['prediction'][..., 1]
        norm = np.maximum(x.max() - x.min(), y.max() - y.min())
        x = (x - x.min()) / norm
        trajectories['prediction'][..., 0] = x + x.max() / 2
        y = (y - y.min()) / norm
        trajectories['prediction'][..., 1] = y + y.max() / 2

        keypoints = (trajectories, timestamps), -1, -1

        event_id, segment_id, start = dataset.clip_infos[dataset.clip_ids[sample_ids[i]]]
        segment, first_frame, last_frame = dataset.trajectories[(event_id, segment_id)]

        root_save = '/path/to/save'
        path_save = Path(f'{root_save}/pca_increment_{opt}/{model_id}/'
                         f'video_delta_{delta}_{i}_{event_id}_{first_frame + start}.mp4')

        dataset.visualize_pose(keypoints, is_video=True, path_save=path_save, blit=True, subplots=True,
                               num_sampled_points=dataset.num_copies, show_axis_=not dataset.clean_viz)
