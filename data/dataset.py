"""
Human Keypoints Dataset
Contains keypoints extracted from following datasets:
    - Dancing videos from Everybody Dance Now (https://carolineec.github.io/everybody_dance_now/)
    - Diving48
    - FisV Figure Skating (https://github.com/loadder/MS_LSTM)
    - FineGym

The keypoint format is in https://cmu-perceptual-computing-lab.github.io/openpose/web/html/doc/md_doc_02_output.html
#pose-output-format-body_25

Not sure why but all the keypoints we extracted give to keypoint 11 the value of keypoint 14 (they are always the same)
For the visualizations we replace the node 11 for the 24

"""
import functools
import json
import multiprocessing
import pickle as pkl
import random
from abc import ABC
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.utils.data
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from moviepy.editor import *
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from utils import utils

reverse_indices = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 11, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]


def plot_skel(keypoints, ax, markersize=10, linewidth=5, alpha=0.7):
    limb_seq = [([17, 15], [238, 0, 255]),
                ([15, 0], [255, 0, 166]),
                ([0, 16], [144, 0, 255]),
                ([16, 18], [65, 0, 255]),
                ([0, 1], [255, 0, 59]),
                ([1, 2], [255, 77, 0]),
                ([2, 3], [247, 155, 0]),
                ([3, 4], [255, 255, 0]),
                ([1, 5], [158, 245, 0]),
                ([5, 6], [93, 255, 0]),
                ([6, 7], [0, 255, 0]),
                ([1, 8], [255, 21, 0]),
                ([8, 9], [6, 255, 0]),
                ([9, 10], [0, 255, 117]),
                # ([10, 11]], [0, 252, 255]),  # See comment above
                ([10, 24], [0, 252, 255]),
                ([8, 12], [0, 140, 255]),
                ([12, 13], [0, 68, 255]),
                ([13, 14], [0, 14, 255]),
                # ([11, 22], [0, 252, 255]),
                # ([11, 24], [0, 252, 255]),
                ([24, 22], [0, 252, 255]),
                ([24, 24], [0, 252, 255]),
                ([22, 23], [0, 252, 255]),
                ([14, 19], [0, 14, 255]),
                ([14, 21], [0, 14, 255]),
                ([19, 20], [0, 14, 255])]

    colors_vertices = {0: limb_seq[4][1],
                       1: limb_seq[11][1],
                       2: limb_seq[5][1],
                       3: limb_seq[6][1],
                       4: limb_seq[7][1],
                       5: limb_seq[8][1],
                       6: limb_seq[9][1],
                       7: limb_seq[10][1],
                       8: limb_seq[11][1],
                       9: limb_seq[12][1],
                       10: limb_seq[13][1],
                       11: limb_seq[14][1],
                       12: limb_seq[15][1],
                       13: limb_seq[16][1],
                       14: limb_seq[17][1],
                       15: limb_seq[1][1],
                       16: limb_seq[2][1],
                       17: limb_seq[0][1],
                       18: limb_seq[3][1],
                       19: limb_seq[21][1],
                       20: limb_seq[23][1],
                       21: limb_seq[22][1],
                       22: limb_seq[18][1],
                       23: limb_seq[20][1],
                       24: limb_seq[19][1]}

    alpha = alpha
    for vertices, color in limb_seq:
        if keypoints[vertices[0]].mean() != 0 and keypoints[vertices[1]].mean() != 0:
            ax.plot([keypoints[vertices[0]][0], keypoints[vertices[1]][0]],
                    [keypoints[vertices[0]][1], keypoints[vertices[1]][1]], linewidth=linewidth,
                    color=[j / 255 for j in color], alpha=alpha)
    # plot kp
    for i in range(len(keypoints)):
        if keypoints[i].mean() != 0:
            ax.plot(keypoints[i][0], keypoints[i][1], 'o', markersize=markersize,
                    color=[j / 255 for j in colors_vertices[i]], alpha=alpha)


def bb_intersection_over_union(boxA, boxB):
    # Code adapted from https://gist.github.com/meyerjo
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = np.maximum(boxA[0], boxB[0])
    yA = np.maximum(boxA[1], boxB[1])
    xB = np.minimum(boxA[2], boxB[2])
    yB = np.minimum(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = np.abs(np.maximum(xB - xA, 0) * np.maximum(yB - yA, 0))

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = np.abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = np.abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / (boxAArea + boxBArea - interArea).astype(float)

    # return the intersection over union value
    return iou


def dist_keypoints(x, y, do_reverse=False):
    """
    Distance function between a pair of keypoints
    """
    x = x[:, None, ...]
    y = y[None, ...]

    if do_reverse:
        y_options = [y, y[..., reverse_indices, :]]
    else:
        y_options = [y]

    distances = []
    for i, y_ in enumerate(y_options):
        # Some body parts are not detected
        keypoints_use = ((x != 0).astype(float) * (y_ != 0).astype(float)).astype(bool).any(-1)

        # The data is [x, y, confidence], we weight by confidence detection
        confidence = x[..., 2] * y_[..., 2]

        dist_point2point = np.sqrt(((x[..., :2] - y_[..., :2]) ** 2).sum(-1))
        dist_point2point = (dist_point2point * confidence * keypoints_use).sum(-1) / (keypoints_use.sum(-1) + 0.01)

        # If fewer than 4 common points, we make the distance very large
        dist_point2point[keypoints_use.sum(-1) < 4] = 10 * dist_point2point.max()

        dist = dist_point2point

        distances.append(dist)

    distances = np.stack(distances)
    dist_best = np.min(distances, 0)
    revers = distances.argmin(0)

    # If the skeleton is small, point2point may be misleading (close but not overlapped). So we add "intersection
    # over union".
    bbox_a = [(x + (x == 0) * 1000).min(-2)[..., 0], (x + (x == 0) * 1000).min(-2)[..., 1],
              (x - (x == 0) * 1000).max(-2)[..., 0], (x - (x == 0) * 1000).max(-2)[..., 1]]
    bbox_b = [(y + (y == 0) * 1000).min(-2)[..., 0], (y + (y == 0) * 1000).min(-2)[..., 1],
              (y - (y == 0) * 1000).max(-2)[..., 0], (y - (y == 0) * 1000).max(-2)[..., 1]]
    iou = bb_intersection_over_union(bbox_a, bbox_b)

    return dist_best, revers, iou


def size_human(keypoints):
    """
    Return the size of the human based on distance between furthest keypoints
    """
    keypoints_use = (keypoints != 0).all(-1)  # Filter out non-detected body parts
    keypoints = keypoints[keypoints_use]
    dist = np.sqrt(((keypoints[None, ...][..., :2] - keypoints[:, None, ...][..., :2]) ** 2).sum(-1))
    size = dist.max()
    return size


def keep_single_trajectory(keypoints, iou_min, boundaries=None):
    """
    Openpose extracts poses for all humans in the video. Frames are computed separately. Here we create "tubelets"
    of trajectories, and then select the most salient trajectories based on heuristics (points for: 1) the largest
    human in the scene, 2) consistent trajectory (small movement), 3) uninterrupted/long trajectory). More than
    one trajectory can be extracted for every video.

    If there is a break in the scene (e.g. camera change), then we treat as separate clips and return different
    trajectories (not overlapping in time)
    """
    start = None
    for i in range(len(keypoints)):
        if keypoints[i] is not None:
            start = i  # Most of the times start should be 0
            break
    if start is None:
        return []  # No keypoints found at all in the video

    last_keypoints_traj = {}
    first_keypoints_traj = {}

    # We don't accumulate tracks, just compute wrt the last frame
    all_trajectories = []
    trajectories = defaultdict(list)  # every element in the list is a frame id and pose id within that frame
    trajectory_values = {}
    for i in range(len(keypoints[start])):
        trajectories[i].append((start, i, False))
        # distance between points, size of human, reverse information
        trajectory_values[i] = [0, size_human(keypoints[start][i]), False]

        last_keypoints_traj[i] = keypoints[start][i]
        first_keypoints_traj[i] = keypoints[start][i]
    old_frame = keypoints[start]

    sizes = np.array([size_human(old_frame[j]) for j in range(len(old_frame))])
    indices = np.where(sizes > (sizes.max() / 4 * 3))[0]
    old_frame = old_frame[indices]

    # assignment from pose_id to id in the trajectories dict
    assignment_old_frame = {i: i for i in range(len(keypoints[start]))}
    mean_assign_dist = None

    frame_boundaries = []
    if boundaries is not None:
        frame_boundaries = [b[1] for b in boundaries]

    for i, frame in enumerate(keypoints[start + 1:]):
        assignment_old_frame_new = {}

        if frame is not None:
            sizes = np.array([size_human(frame[j]) for j in range(len(frame))])
            indices = np.where(sizes > (sizes.max() / 4 * 3))[0]
            frame = frame[indices]

        if frame is not None and old_frame is not None:
            dist_matrix, revers, iou = dist_keypoints(old_frame, frame, do_reverse=True)

            # Hungarian may fail when there is a skeleton without a pair, it may assign an incorrect one (and that one
            # will not be assigned to its correct one) if the sum of the two incorrectness is smaller than just one
            # incorrectness. To avoid that, we pre-filter matches that are implausible given distance
            if mean_assign_dist is not None:
                mask = np.logical_or(dist_matrix > 10 * mean_assign_dist, iou < iou_min)
                dist_matrix_ = dist_matrix + mask * 100
            else:
                dist_matrix_ = dist_matrix

            row_ind, col_ind = linear_sum_assignment(dist_matrix_)  # hungarian algorithm

            if mean_assign_dist is None:
                mean_assign_dist = dist_matrix[row_ind, col_ind].mean()
            else:
                mean_assign_dist = mean_assign_dist * 0.9 + dist_matrix[row_ind, col_ind].mean() * 0.1

            if i + start in frame_boundaries:
                row_ind = col_ind = []

            for j in range(len(row_ind)):  # For every element in the past
                dist = dist_matrix[row_ind[j], col_ind[j]]
                iou_ = iou[row_ind[j], col_ind[j]]
                rev = revers[row_ind[j], col_ind[j]]
                traj_idx = assignment_old_frame[row_ind[j]]
                # otherwise, finish trajectory
                if dist <= 10 * mean_assign_dist and iou_ >= iou_min:
                    trajectory_values[traj_idx][0] = (trajectory_values[traj_idx][0] * len(trajectories[traj_idx]) +
                                                      dist) / (len(trajectories[traj_idx]) + 1)
                    size_hum = size_human(frame[col_ind[j]])
                    trajectory_values[traj_idx][1] = (trajectory_values[traj_idx][1] * len(trajectories[traj_idx]) +
                                                      size_hum) / (len(trajectories[traj_idx]) + 1)

                    # Will need to be reversed if the previous one was "correct" and this is reversed wrt the previous
                    # one, or if the last one was "incorrect" and this is the same as the previous one
                    last_reverse = trajectory_values[traj_idx][2]
                    needs_reverse = rev if not last_reverse else not rev

                    trajectories[traj_idx].append((i + start + 1, indices[col_ind[j]], needs_reverse))
                    trajectory_values[traj_idx][2] = needs_reverse

                    assignment_old_frame_new[col_ind[j]] = traj_idx

                    last_keypoints_traj[traj_idx] = frame[col_ind[j]]

        # Spawn new trajectories
        if frame is not None:
            for j in range(frame.shape[0]):
                if j not in assignment_old_frame_new:
                    traj_idx = len(trajectories)
                    trajectories[traj_idx].append((i + start + 1, indices[j], False))
                    trajectory_values[traj_idx] = [0, size_human(frame[j]), False]
                    assignment_old_frame_new[j] = traj_idx
                    first_keypoints_traj[traj_idx] = frame[j]
                    last_keypoints_traj[traj_idx] = frame[j]

        old_frame = frame
        assignment_old_frame = assignment_old_frame_new

    # Stack trajectories that got broken. Maximum 1 step (for now)
    to_concat = {}
    for k, v in trajectories.items():
        if len(v) == 1:  # I think it adds more noise than anything.
            continue
        start_trajectory = v[0][0]
        if start_trajectory in frame_boundaries:
            continue  # not to be linked with any past trajectory
        for k2, v2 in trajectories.items():
            if len(v2) == 1:
                continue
            end_trajectory = v2[-1][0]
            if start_trajectory == end_trajectory + 2 or start_trajectory == end_trajectory + 1:  # one was skipped
                # compute distance
                dist, revers_, iou_ = dist_keypoints(last_keypoints_traj[k2][None, :], first_keypoints_traj[k][None, :],
                                                     do_reverse=True)
                dist, revers_, iou_ = dist[0, 0], revers_[0, 0], iou_[0, 0]

                if dist <= 10 * mean_assign_dist and iou_ >= iou_min:
                    gap = start_trajectory == end_trajectory + 2
                    to_concat[k] = (k2, revers_, gap)

    # We iterate across "future" of trajectory, and add to "past". That "past" may have been the future of something
    # else before, so we have to go and look for the segment that is most at the beginning
    for k, v in trajectories.items():
        k2 = k
        rev, gap = None, None
        while k2 in to_concat:
            k2, rev, gap = to_concat[k2]
        if k2 != k:
            if rev:
                assert not v[0][2]  # First in sequence has to be False
                v[0] = (v[0][0], v[0][1], True)
            if gap:  # There is a 1-frame gap:
                trajectories[k2] = trajectories[k2] + [(None, None, None)] + v
            else:
                trajectories[k2] = trajectories[k2] + v
            # trajectory_values[.][2] is not used more later
            trajectory_values[k2][0] = (trajectory_values[k2][0] * len(trajectories[k2]) +
                                        trajectory_values[k][0] * len(trajectories[k])) / \
                                       (len(trajectories[k2]) + len(trajectories[k]))
            trajectory_values[k2][1] = (trajectory_values[k2][1] * len(trajectories[k2]) +
                                        trajectory_values[k][1] * len(trajectories[k])) / \
                                       (len(trajectories[k2]) + len(trajectories[k]))
            trajectories[k] = []

    single_trajectories = []

    all_sizes = np.array([v[1] for v in trajectory_values.values()])
    all_distances = np.array([v[0] for k, v in trajectory_values.items() if len(trajectories[k]) > 1])

    # for i, (trajectories, trajectory_values) in enumerate(zip(all_trajectories, all_trajectories_values)):
    for k, traj_vals in trajectory_values.items():
        if len(trajectories[k]) > 1:
            len_traj = len(trajectories[k])

            # Size is more restrictive than distance. A lot of small people not important. Also, distances already
            # good most of them because trajectories created based on distances
            # We also limit minimum movement (relative to body size) to not get static trajectories

            # if len_traj > 10:
            if len_traj > 10 and traj_vals[1] > np.percentile(all_sizes, 90) and \
                    np.percentile(all_distances, 90) > traj_vals[0] > 0.01 * traj_vals[1]:

                single_trajectory = []
                for frame_idx, pose_idx, needs_reverse in trajectories[k]:
                    if frame_idx is None:
                        keyp = single_trajectory[-1]  # Copy previous one
                    else:
                        keyp = keypoints[frame_idx][pose_idx]
                    if needs_reverse:
                        keyp = keyp[reverse_indices]
                    single_trajectory.append(keyp)

                start_frame_idx = trajectories[k][0][0]
                end_frame_idx = trajectories[k][-1][0]
                single_trajectory = (np.stack(single_trajectory), start_frame_idx, end_frame_idx)
                single_trajectories.append(single_trajectory)

    return single_trajectories


class HumanKeypointsDataset(torch.utils.data.Dataset, ABC):
    def __init__(self,
                 split,  # train, validate, test, all
                 path=None,  # Dataset path
                 dataset_name='',  # Name of the dataset
                 max_clips_split=None,  # Maximum size of the split
                 proportion_samples=None,  # Just use a proportion of the dataset samples
                 select_indices=None,  # Select specific indices for the dataset
                 info_all_splits=None,  # Precomputed information for all splits
                 restart_samples=False,  # Recompute information (in case we obtain new data)
                 max_steps=6,  # Maximum number of steps sampled from a trajectory
                 min_steps_past=3,  # Minimum number of steps sampled from the past segment
                 min_steps_future=3,  # Minimum number of steps sampled from the future segment
                 pca_augmentation=None,  # In case we need to do some augmentation of the data
                 extrapolate_future=False,  # Just for inference time
                 just_future=False,  # Just for inference time
                 predict_interpolate=False,  # Just for inference time
                 uniform=False,  # Sample uniformly
                 first_samples=False,  # Sample first samples
                 uniform_interpolate=False,  # Sample uniformly
                 num_copies=10,  # For PCA visualizations
                 temporal_noise=True,  # Add noise to the temporal steps
                 invert_time=False,  # Invert time indices. Make sequence go backwards
                 invert_time_rnn=False,  # Make sequence go backwards, but invert data points instead of times
                 clean_viz=False,  # Visualize without axis or titles
                 all_together=False,  # Visualize all trajectories on top of each other
                 seed=42,  # Random noise seed
                 **kwargs):

        assert split in ["train", "validate", "test", "all"], f"Split '{split}' not supported."
        assert max_steps >= min_steps_past + min_steps_future

        self.split = split
        self.dataset_path = Path(path)
        self.dataset_name = dataset_name
        self.max_clips_split = max_clips_split
        self.proportion_samples = proportion_samples
        self.select_indices = select_indices
        self.max_steps = max_steps
        self.min_steps_past = min_steps_past
        self.min_steps_future = min_steps_future
        self.pca_augmentation = pca_augmentation
        self.extrapolate_future = extrapolate_future
        self.just_future = just_future
        self.predict_interpolate = predict_interpolate
        self.uniform = uniform
        self.first_samples = first_samples
        self.uniform_interpolate = uniform_interpolate
        self.num_copies = num_copies
        self.temporal_noise = temporal_noise
        self.invert_time = invert_time
        self.invert_time_rnn = invert_time_rnn
        self.clean_viz = clean_viz
        self.all_together = all_together
        self.seed = seed

        if self.split in ['train', 'validate']:
            # -1 indicates random sampling.
            self.temporal_sample_index = -1
            self.spatial_sample_index = -1

        else:  # test or all
            self.temporal_sample_index = 1  # middle
            self.spatial_sample_index = 1  # middle

        self.clip_infos = None
        self.clip_ids = None

        self.proportions = {'train': 0.8, 'validate': 0.1, 'all': 1.}
        self.proportions['test'] = 1 - self.proportions['train'] - self.proportions['validate']

        if info_all_splits is None:
            path_clip_infos = self.dataset_path / self.name_info
            if restart_samples or not os.path.isfile(path_clip_infos):
                clip_infos, *other_info = self.prepare_samples()
                print('Saving dataset')
                torch.save([clip_infos, *other_info], path_clip_infos)
            else:
                print('Loading dataset')
                clip_infos, *other_info = torch.load(path_clip_infos)
                print('Dataset loaded')
            clip_ids = list(clip_infos.keys())
            last_i = 0
            clip_ids_splits = {}
            for split_ in self.proportions.keys():
                start = 0 if split_ == 'all' else last_i
                end = start + int(np.round(self.proportions[split_] * len(clip_ids)))
                end = np.minimum(end, len(clip_ids))
                if split_ == 'all':
                    start = 0
                else:
                    last_i = end
                clip_ids_splits[split_] = [clip_ids[i] for i in range(start, end)]
                if self.max_clips_split is not None and len(clip_ids_splits[split_]) > self.max_clips_split:
                    # Mostly for visualization or debugging purposes
                    # Random so that not all clips belong to the same video
                    clip_ids_splits[split_] = [clip_ids_splits[split_][i] for i in
                                               random.sample(range(len(clip_ids_splits[split_])), self.max_clips_split)]
            info_all_splits = [clip_infos, clip_ids_splits, other_info]

        self.info_all_splits = info_all_splits

        self.clip_infos = info_all_splits[0]
        if self.split == 'all':
            self.clip_ids = info_all_splits[1]['train'] + info_all_splits[1]['validate'] + info_all_splits[1]['test']
        else:
            self.clip_ids = info_all_splits[1][split]

        if not hasattr(self, 'max_steps'):
            self.max_steps = None

        self.trajectories, = self.info_all_splits[2][0]

    @property
    def save_info_path(self):
        """
        Path to save info about the dataset
        """
        info_path = self.dataset_path / 'data_info'
        os.makedirs(info_path, exist_ok=True)
        return info_path

    @property
    def name_info(self):
        return f'clip_infos_{self.max_steps}.pth'

    @classmethod
    def spatial_crop(cls, x, reference):
        """
        x: [B, T, (S), 50]
        reference: [B, T, 50]
        Spatially crops x following the spatial crops of the reference. Keeps the gradient on x
        """
        x = x.view(*x.shape[:-1], 25, 2)
        reference = reference.view(*reference.shape[:-1], 25, 2)
        indices_zero = reference.sum(-1) == 0  # [B, T, 25]
        if len(x.shape) == 5:  # multiple samples
            indices_zero = indices_zero.unsqueeze(2)
        indices_zero = indices_zero.unsqueeze(-1)  # Add the spatial dimension
        x_ = x * (1 - indices_zero.float())

        indices_hundred = reference == -100  # [B, T, 25]
        if len(x.shape) == 5:  # multiple samples
            indices_hundred = indices_hundred.unsqueeze(2)
        x_ = x_ * (1 - indices_hundred.float()) - 100 * indices_hundred.float()

        return x_.view(*x.shape[:-2], 50)

    def prepare_sequence(self, positions, time_indices, temporal_noise=False, noise_seed=None, max_steps=None):
        """
        Move to tensor and zero-pad if necessary
        We pad with -100, to have an easy-to-debug value. But note that -100 can still be an acceptable value, we will
        never filter by -100; we need to use the mask all the time.
        An alternative would be to use NaN, but they are very bad at backpropagating properly.

        If temporal_noise, instead of having integer values for time indices,
        """
        assert self.max_steps is not None, 'Only for datasets where we have max_steps'

        if type(positions) != torch.Tensor and positions is not None:
            positions = torch.tensor(positions).float()
        if type(time_indices) != torch.Tensor:
            time_indices = torch.tensor(time_indices).float()
        if positions is not None:
            assert time_indices.shape[0] == positions.shape[0]

        duration = time_indices.shape[0]

        # None of the past/future alone will be as long as self.max_steps, but we give room for inputting whole
        # sequences
        if max_steps is None:
            max_steps = self.max_steps  # if not self.extrapolate_future else 2*self.max_steps
        if duration < max_steps:
            if positions is not None:
                padding_pos = torch.zeros((max_steps - duration, *positions.shape[1:])).to(positions.device)
                positions = torch.cat([positions, padding_pos - 100])
            padding_time = torch.zeros(max_steps - duration).to(time_indices.device)
            time_indices = torch.cat([time_indices, padding_time - 100])
        elif duration > max_steps:
            positions = positions[:max_steps] if positions is not None else None
            time_indices = time_indices[:max_steps]

        if temporal_noise:
            # Make the sequence start a bit later (so relative increments are the same)
            time_indices[time_indices != -100] += utils.str_to_probability(noise_seed)

        return positions, time_indices, duration

    @staticmethod
    def visualize_pose(keypoints, img_path=None, is_video=False, path_save=None, blit=False, subplots=False,
                       num_sampled_points=1, show_axis_=True, all_together_=False):
        """
        :param all_together_:
        :param keypoints:
        :param img_path:
        :param is_video: is video or image
        :param path_save:
        :param blit: if we do not want to update the drawing at every frame.
            https://alexgude.com/blog/matplotlib-blitting-supernova/
        :param subplots: several trajectories, so we draw in subplots
        :param num_sampled_points: how many samples are sampled from distribution
        :param show_axis_: show axis and title
        :return:
        """

        dpi = 30 if show_axis_ and not is_video else 300  # We do not show axis for nice visualizations
        num_samples = None
        if subplots:
            # num_samples = np.minimum(5, num_sampled_points)
            num_samples = num_sampled_points
            base_size = 3
            num_rows = len(keypoints[0][0])
            fig, axes = plt.subplots(num_rows, 1 if all_together_ else num_samples, dpi=dpi,
                                     figsize=(base_size * (1 if all_together_ else num_samples), base_size * num_rows))
            fig.tight_layout()
            if type(axes) != np.ndarray:
                axes = np.array([axes])
            if len(axes.shape) == 1:
                if num_rows == 1:
                    axes = axes[None, :]
                else:
                    axes = axes[:, None]
        else:
            fig = plt.figure(dpi=dpi)
            axes = np.array([[plt.gca()]])

        if is_video:
            if type(keypoints) == tuple:
                keypoints, start_frame_idx, end_frame_idx = keypoints
            else:
                start_frame_idx = end_frame_idx = None
            timestamps = name_steps = None
            if blit:
                keypoints, (timestamps, name_steps) = keypoints
            if not subplots:
                keypoints = {'trajectory': keypoints}
            if img_path is not None:
                video = VideoFileClip(str(img_path))
                fps = video.reader.fps
                if end_frame_idx is not None:
                    clip = video.subclip(start_frame_idx / fps, end_frame_idx / fps)
                else:
                    clip = video
            else:
                fps = 25
                clip = None

            path_save.parent.mkdir(parents=True, exist_ok=True)
            path_dir = (path_save.parent / path_save.stem)
            path_dir.mkdir(parents=True, exist_ok=True)

            next_keypoint: int = 0

            def animate(j):
                nonlocal next_keypoint
                nonlocal path_dir
                time_frame = j / fps
                if blit and (len(timestamps) <= next_keypoint or timestamps[next_keypoint] > time_frame):
                    return ()  # do not update figure

                if blit:
                    i = next_keypoint
                else:
                    i = j

                next_keypoint += 1

                for ax in axes.reshape(-1):
                    ax.axis('off')
                    ax.axes.xaxis.set_visible(False)
                    ax.axes.yaxis.set_visible(False)

                for row, (name, trajectory) in enumerate(keypoints.items()):
                    traj_limits = trajectory[trajectory != -100]  # Used to compute limits
                    if len(trajectory.shape) == 3:
                        trajectory = trajectory[:, None]  # Simulate multiple samples

                    for s in range(trajectory.shape[1]):
                        if ((num_samples is not None) and (s >= num_samples)) or (trajectory[i][s] == -100).any():
                            break  # Above we have a limit of S=num_samples columns
                        if type(axes) == np.ndarray:
                            ax = axes[row][0 if all_together_ else s]
                        else:
                            ax = axes  # no subplots
                        if not (all_together_ and s > 0):
                            ax.clear()
                        if img_path is not None:
                            frame = clip.get_frame(i / fps)
                            ax.imshow(frame)
                        else:  # Make uniform across video
                            margin = (traj_limits.max() - traj_limits.min()) * 0.1
                            ax.set_xlim([traj_limits.min() - margin, traj_limits.max() + margin])
                            ax.set_ylim([traj_limits.max() + margin, traj_limits.min() - margin])
                        if len(trajectory[i].shape) == 3:
                            # We would do this if we want all samples superimposed
                            # for j in range(len(trajectory[i])):
                            #     plot_skel(trajectory[i][j], ax, markersize=5, linewidth=3)
                            plot_skel(trajectory[i][s], ax, markersize=5, linewidth=3,
                                      alpha=0.3 if all_together_ else 0.7)
                        else:
                            plot_skel(trajectory[i][s], ax)
                        if show_axis_:
                            ax.axis('on')  # View the bounds of the axis
                        else:
                            ax.axis('off')
                        if subplots and s == 0 and show_axis_:
                            if name_steps is not None:
                                title = name_steps[i] + ' from ' + name
                            else:
                                title = name
                            ax.set_title(title)

                if not subplots:
                    plt.margins(0, 0)
                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

                # plt.savefig(path_dir / f'{j}.pdf')
                return fig,  # unused if not blit

            def init_func():
                for ax in axes.reshape(-1):
                    ax.axis('off')
                    ax.axes.xaxis.set_visible(False)
                    ax.axes.yaxis.set_visible(False)
                return fig,

            num_frames_plot = int(timestamps[-1] * 1 * fps) + 1 if blit else len(keypoints['trajectory'])
            ani = FuncAnimation(fig, animate, num_frames_plot, repeat=False, interval=1000 / fps, blit=blit,
                                init_func=init_func)
            ani.save(path_save)

        else:  # image
            if img_path is not None:
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.imshow(img)
            else:
                plt.gca().invert_yaxis()
            plot_skel(keypoints, plt.gca())
            plt.axis('off')
            if path_save is not None:
                plt.savefig(path_save, bbox_inches='tight', pad_inches=0)
            else:
                # Return as numpy array
                fig.canvas.draw()
                image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                return image_from_plot
        plt.close()
        plt.cla()
        plt.clf()

    def visualize_trajectories(self, saved_tensors, save_names, reconstruct_intersection, model_id, vrnn_model,
                               **kwargs):
        extra_info = ''
        show_axis = not self.clean_viz
        if vrnn_model:
            sample_ids, time_indices_past, time_indices_future, video_len_past, video_len_future, seg_inp_p, \
            seg_inp_f, y = saved_tensors
            num_sampled_points = 1
            seg_inp_p_ = seg_inp_p
            if len(y.shape) == 4:
                num_sampled_points = y.shape[2]
                seg_inp_p_ = seg_inp_p.unsqueeze(2).expand_as(y)
            times = [('past', time_indices_past, video_len_past), ('future', time_indices_future, video_len_future)]
            options = [('ground truth', seg_inp_p, seg_inp_f), ('gt past, pred future', seg_inp_p_, y)]
        else:
            d = {save_names[i]: saved_tensors[i] for i in range(len(save_names))}
            sample_ids = d['sample_ids']
            num_sampled_points = d['seg_dec_p_inp_p'].shape[2]

            if self.extrapolate_future:
                extra_info = 'extrapolate_'
                times = [('past', d['time_indices_past'], d['video_len_past']),
                         ('extrapolate', d['time_indices_decode'], d['video_len_decode'])]
                options = [('ground truth', d['seg_inp_p'], None),
                           ('past', d['seg_dec_p_inp_p'], d['seg_dec_decode_inp_p']),
                           ('intersection', d['seg_dec_p_inp_p__inter__inp_f'],
                            d['seg_dec_decode_inp_p__inter__inp_f']),
                           ('from all', d['seg_dec_p_inp_a'], d['seg_dec_decode_inp_a'])]

            elif self.just_future:
                extra_info = 'website_'
                times = [('past', d['time_indices_past_decode'], d['video_len_past_decode']),
                         ('future', d['time_indices_future_decode'], d['video_len_future_decode'])]
                options = [('past', d['seg_dec_p_inp_p'], d['seg_dec_f_inp_p'])]

            elif self.predict_interpolate:

                # Adapt ground truth to continuous format
                gt_past_orig = d['seg_inp_p']
                gt_future_orig = d['seg_inp_f']
                aux_1_past = d['time_indices_past_decode'][..., None] <= d['time_indices_past'][:, None, :]
                aux_2_past = aux_1_past.float().argmax(dim=-1)  # Argmax returns the first index
                aux_1_future = d['time_indices_future_decode'][..., None] <= d['time_indices_future'][:, None, :]
                aux_2_future = aux_1_future.float().argmax(dim=-1)  # Argmax returns the first index
                gt_past = gt_past_orig[torch.arange(gt_past_orig.shape[0]).unsqueeze(1), aux_2_past]
                gt_future = gt_future_orig[torch.arange(gt_future_orig.shape[0]).unsqueeze(1), aux_2_future]

                times = [('past', d['time_indices_past_decode'], d['video_len_past_decode']),
                         ('interpolate', d['time_indices_decode'], d['video_len_decode']),
                         ('future', d['time_indices_future_decode'], d['video_len_future_decode'])]

                if self.clean_viz:
                    extra_info = 'interpolate_clean_'
                    options = [('intersection', d['seg_dec_p_inp_p__inter__inp_f'],
                                d['seg_dec_decode_inp_p__inter__inp_f'], d['seg_dec_f_inp_p__inter__inp_f'])]
                else:
                    extra_info = 'interpolate_'
                    options = [('ground truth', gt_past, None, gt_future),
                               ('past', d['seg_dec_p_inp_p'], d['seg_dec_decode_inp_p'], d['seg_dec_f_inp_p']),
                               ('intersection', d['seg_dec_p_inp_p__inter__inp_f'],
                                d['seg_dec_decode_inp_p__inter__inp_f'], d['seg_dec_f_inp_p__inter__inp_f'])]

            else:  # Regular case
                times = [('past', d['time_indices_past'], d['video_len_past']),
                         ('future', d['time_indices_future'], d['video_len_future'])]
                options = [('ground truth', d['seg_inp_p'], d['seg_inp_f']),
                           ('past', d['seg_dec_p_inp_p'], d['seg_dec_f_inp_p']),
                           ('future', d['seg_dec_p_inp_f'], d['seg_dec_f_inp_f'])]
                if reconstruct_intersection:
                    options.append(('from intersection', d['seg_dec_p_inp_p__inter__inp_f'],
                                    d['seg_dec_f_inp_p__inter__inp_f']))

        for i in range(1024):

            if len(options) == 0:
                continue
            trajectories = {}
            for option in options:
                name, *tensors = option
                traj = []
                for k, tensor in enumerate(tensors):
                    if tensor is None:
                        traj.append(-100 * torch.ones([times[k][2][i], 50]))
                    else:
                        traj.append(tensor[i, :times[k][2][i]])
                traj = torch.cat(traj)
                traj = traj.view(*traj.shape[:-1], 25, 2)
                trajectories[name] = traj

            timestamps = torch.cat([times[k][1][i][:times[k][2][i]] for k in range(len(times))])
            name_steps = [ss for s in [[times[k][0]] * times[k][2][i] for k in range(len(times))] for ss in s]

            timestamps = timestamps - timestamps[0]  # First timestamp start at 0
            timestamps = timestamps / 25  # fps = 25
            timestamps = (timestamps, name_steps)
            keypoints = (trajectories, timestamps), -1, -1

            # Get segment info
            event_id, segment_id, start = self.clip_infos[self.clip_ids[sample_ids[i]]]
            segment, first_frame, last_frame = self.trajectories[(event_id, segment_id)]

            path_save = Path(f'/path/to/save/videos/{model_id}/'
                             f'video_{extra_info}{i}_{event_id}_{first_frame + start}.mp4')

            self.visualize_pose(keypoints, is_video=True, path_save=path_save, blit=True, subplots=True,
                                num_sampled_points=num_sampled_points, show_axis_=show_axis,
                                all_together_=self.all_together)

    def visualize_sample(self, index, path_save=None):
        event_id, segment_id, *_ = self.clip_infos[self.clip_ids[index]]
        segment = self.trajectories[(event_id, segment_id)]
        if path_save is None:
            path_save = Path(f'/path/to/save/datasets/{self.dataset_name}/examples/'
                             f'viz_{event_id}_{segment[1]}_{segment[2]}.mp4')
        path_save.parent.mkdir(parents=True, exist_ok=True)
        if not path_save.exists():
            self.visualize_pose(segment, is_video=True, path_save=path_save, img_path=self.get_video_path(event_id))

    def augment_trajectories(self, original, j):
        if self.pca_augmentation == 'spatial_flip' and j == 1:  # j == 0 is the no-flip sample
            o = deepcopy(original.view(-1, 25, 2))
            o_h, o_v = o[..., 0], o[..., 1]
            mask = torch.logical_and(o_h != -1, o_h != -100)
            o_h[mask] = 1 - o_h[mask]
            new_traj = torch.stack([o_h, o_v], dim=-1).view(-1, 50)
            return new_traj
        else:
            return original

    def augment_time_inputs(self, original, j, total_j):
        if self.pca_augmentation == 'speed':
            # From half the speed (twice the duration) to twice the speed (half the duration)
            new_traj = deepcopy(original)
            new_traj[new_traj != -100] = new_traj[new_traj != -100] * (j * (2 - 0.5) / (total_j - 1) + 0.5)
            return new_traj
        elif self.pca_augmentation == 'temporal_offset':
            # Start from 0 to total_j
            new_traj = deepcopy(original)
            new_traj[original != -100] = original[original != -100] + j
            return new_traj
        else:
            return original

    def get_segments(self, event, path_keypoints, max_steps, iou_min):
        trajectories = []
        sub_event_id = 0
        event_id = event.replace('.pkl', '')
        shot_transitions = None
        if (self.dataset_path / 'shot_transitions.pth').exists():
            shot_transitions = torch.load(self.dataset_path / 'shot_transitions.pth')
        if event.endswith('.pkl'):
            with open(os.path.join(path_keypoints, event), 'rb') as f:
                data_keypoint = pkl.load(f)

            boundaries = None
            if shot_transitions is not None:
                boundaries = shot_transitions[event.split('.')[0]]
            # For every trajectory, we create segments. We can still create sub-segments within those segments
            # later, so not everything is fixed here (although we could fix it)
            segments = keep_single_trajectory(data_keypoint, iou_min, boundaries)

            for segment in segments:
                if len(segment[0]) >= 3 * max_steps:  # We do not sample uniform consecutive steps, need some gap
                    trajectories.append((event_id, sub_event_id, segment))
                    sub_event_id += 1

        return trajectories

    def prepare_samples(self):
        path_keypoints = self.dataset_path / 'keypoints'
        trajectories = {}
        clip_infos = {}
        list_paths = os.listdir(path_keypoints)
        list_paths = [p for p in list_paths if not p.startswith('.')]

        with multiprocessing.Pool(processes=50) as pool:
            results = list(tqdm(pool.imap(functools.partial(self.get_segments, path_keypoints=path_keypoints,
                                                            max_steps=self.max_steps, iou_min=self.iou_min),
                                          list_paths),
                                total=len(list_paths), desc='Precomputing subclip information'))

        total_clips = 0
        for even_results in results:
            for segments in even_results:
                event_id, sub_event_id, segment = segments
                keypoints, start_frame_idx, end_frame_idx = segment
                trajectories[(event_id, sub_event_id)] = segment
                for start in range(len(keypoints) - 3 * self.max_steps):
                    clip_infos[total_clips] = (event_id, sub_event_id, start)
                    total_clips += 1

        return clip_infos, (trajectories,)

    def load_segment(self, index):
        event_id, segment_id, *start = self.clip_infos[self.clip_ids[index]]
        segment = self.trajectories[(event_id, segment_id)]
        if type(segment) == tuple:
            segment = segment[0]  # The rest of elements in the tuple are the first and last frames of the segment
        segment = segment[..., :2]
        if len(start) > 0:
            start = start[0]
            segment = segment[start:start + 3 * self.max_steps]
        return segment

    def __getitem__(self, index):

        if self.select_indices is not None:
            index = self.select_indices[index]

        state = None
        if self.split == 'test':
            state = random.getstate()
            random.seed(index + self.seed)

        segment = self.load_segment(index)

        # Normalize
        # We assume no existing joint will have the two coordinates equal to the minimum coordinate. Otherwise, that
        # would go to [0, 0] as if it did not exist.
        min_norm = segment[segment != 0].min()
        max_norm = segment.max()
        segment = (segment - min_norm) / (max_norm - min_norm)
        # For some reason segment[segment !=0] =  segment[segment !=0] ... did not work
        segment = np.maximum(segment, 0)

        if self.predict_interpolate:  # We need to make room for the interpolation prediction
            steps_total = self.max_steps
            len_past = self.min_steps_past
            len_future = self.min_steps_future
        elif self.just_future:
            steps_total = self.max_steps
            len_past = self.min_steps_past
            len_future = self.max_steps - self.min_steps_past
        else:
            # if self.split == 'test':  # Make reproducible
            #     steps_total = self.max_steps
            #     len_past = self.min_steps_past + (self.max_steps - self.min_steps_past - self.min_steps_future) // 2
            # else:
            steps_total = random.randint(self.min_steps_past + self.min_steps_future, self.max_steps)
            len_past = random.randint(self.min_steps_past, steps_total - self.min_steps_future)
            len_future = steps_total - len_past

        samples_consider = 3 * steps_total  # len(segment) should always be >= to this
        if self.uniform:
            step = 3  # Same number than two lines above
            samples_indices = np.arange(0, samples_consider, step)
            assert len(samples_indices) == steps_total
        elif self.first_samples:
            # This only samples the first samples for the past. The future (and intersection) are random
            samples_past = list(range(len_past))
            samples_other = random.sample(range(len_past, samples_consider), steps_total - len_past)
            samples_other.sort()
            samples_indices = samples_past + samples_other
        else:
            samples_indices = random.sample(range(samples_consider), steps_total)
            samples_indices.sort()
        samples_indices = np.array(samples_indices)
        sub_segment_start = random.randint(0, len(segment) - samples_consider)

        time_indices_past = samples_indices[:len_past]
        time_indices_future = samples_indices[-len_future:]

        positions_past = segment[sub_segment_start + time_indices_past]
        positions_future = segment[sub_segment_start + time_indices_future]

        if self.invert_time:
            min_time = time_indices_past.min()
            max_time = time_indices_future.max()
            time_indices_past = min_time + max_time - time_indices_past
            time_indices_future = min_time + max_time - time_indices_future

        elif self.invert_time_rnn:
            # Time indices are not used
            aux_time = time_indices_past
            aux_pos = positions_past
            time_indices_past = time_indices_future[::-1].copy()
            time_indices_future = aux_time[::-1].copy()
            positions_past = positions_future[::-1].copy()
            positions_future = aux_pos[::-1].copy()

        positions_past, time_indices_past, duration_past = \
            self.prepare_sequence(positions_past, time_indices_past, temporal_noise=self.temporal_noise,
                                  noise_seed=index)
        positions_future, time_indices_future, duration_future = \
            self.prepare_sequence(positions_future, time_indices_future, temporal_noise=self.temporal_noise,
                                  noise_seed=index)

        dict_return = dict(past=positions_past.view(-1, 50), future=positions_future.view(-1, 50),
                           video_len=steps_total, video_len_past=duration_past, video_len_future=duration_future,
                           time_indices_past=time_indices_past, time_indices_future=time_indices_future, index=index,
                           normalize_info=np.stack([min_norm, max_norm]))

        # Also return "all" (the whole segment).
        # We could just concatenate past and future, but to avoid shortcuts, the temporal steps that we sample are
        # different from the ones sampled in past and future. Because steps_total is prior to samples_consider, we can
        # reuse samples_consider.

        if self.uniform:
            step = 3
            samples_indices_all = np.arange(0, samples_consider, step)
            assert len(samples_indices) == steps_total
        else:
            samples_indices_all = random.sample(range(samples_consider), steps_total)
            samples_indices_all.sort()

        samples_indices_all = np.array(samples_indices_all)
        positions_all = segment[sub_segment_start + samples_indices_all]
        positions_all, time_indices_all, duration_all = \
            self.prepare_sequence(positions_all, samples_indices_all, temporal_noise=self.temporal_noise,
                                  noise_seed=index)

        dict_return['all'] = positions_all.view(-1, 50)
        dict_return['time_indices_all'] = time_indices_all

        if self.extrapolate_future:
            time_indices_decode = torch.cat([time_indices_all[:duration_all],
                                             time_indices_all[duration_all - 1] + 1 + time_indices_all[:duration_all]])
            padding_time = torch.zeros(2 * self.max_steps - time_indices_decode.shape[0])
            time_indices_decode = torch.cat([time_indices_decode, padding_time - 100])
            dict_return['time_indices_decode'] = time_indices_decode
            dict_return['video_len_decode'] = 2 * duration_all

        elif self.just_future:
            step = 1
            max_steps = samples_consider

            min_past = time_indices_past[:len_past].min().floor().int()
            max_past = time_indices_past[:len_past].max().floor().int()
            min_future = max_past
            max_future = time_indices_future[:len_future].max().floor().int()

            dict_return['future'] = dict_return['future'] * 0 + 0.1  # Make sure it's not being used
            dict_return['all'] = dict_return['all'] * 0 + 0.1

            continuous_times_past = torch.arange(min_past, max_past, step).numpy()
            continuous_times_future = torch.arange(min_future, max_future, step).numpy()

            _, dict_return['time_indices_past_decode'], dict_return['video_len_past_decode'] = \
                self.prepare_sequence(None, continuous_times_past, temporal_noise=self.temporal_noise,
                                      noise_seed=index, max_steps=max_steps)
            _, dict_return['time_indices_future_decode'], dict_return['video_len_future_decode'] = \
                self.prepare_sequence(None, continuous_times_future, temporal_noise=self.temporal_noise,
                                      noise_seed=index, max_steps=max_steps)

        elif self.predict_interpolate:
            min_past = time_indices_past[:len_past].min().floor().int()
            max_past = time_indices_past[:len_past].max().floor().int()
            min_future = time_indices_future[:len_future].min().floor().int()
            max_future = time_indices_future[:len_future].max().floor().int()

            """
            This can be a little bit counter-intuitive: if self.uniform_interpolate, we replicate conditions where the 
            sampling is uniform with a uniform gap between samples, where the gap is usually not 1 (in this dataset it 
            is actually 3, the value of "step"). If not self.uniform_interpolate, then we choose to sample densely 
            ("uniform" with gap 1).
            """
            if self.uniform_interpolate:
                step = 3
                max_steps = steps_total
            else:
                step = 1
                max_steps = samples_consider

            continuous_times_past = torch.arange(min_past, max_past, step).numpy()
            continuous_times_future = torch.arange(min_future, max_future, step).numpy()
            continuous_times_all = torch.arange(min_past, max_future, step).numpy()
            continuous_times_intersection = torch.arange(max_past + 1, min_future, step).numpy()

            if self.uniform_interpolate and len(continuous_times_intersection) + len_past > steps_total:
                continuous_times_intersection = continuous_times_intersection[:steps_total - len_past]

            positions_interpolate = segment[continuous_times_intersection]

            _, dict_return['time_indices_past_decode'], dict_return['video_len_past_decode'] = \
                self.prepare_sequence(None, continuous_times_past, temporal_noise=self.temporal_noise,
                                      noise_seed=index, max_steps=max_steps)
            _, dict_return['time_indices_future_decode'], dict_return['video_len_future_decode'] = \
                self.prepare_sequence(None, continuous_times_future, temporal_noise=self.temporal_noise,
                                      noise_seed=index, max_steps=max_steps)
            _, dict_return['time_indices_all_decode'], dict_return['video_len_all_decode'] = \
                self.prepare_sequence(None, continuous_times_all, temporal_noise=self.temporal_noise,
                                      noise_seed=index, max_steps=max_steps)
            positions_interpolation, dict_return['time_indices_decode'], dict_return['video_len_decode'] = \
                self.prepare_sequence(positions_interpolate, continuous_times_intersection,
                                      temporal_noise=self.temporal_noise, noise_seed=index, max_steps=max_steps)
            dict_return['interpolation'] = positions_interpolation.view(-1, 50)

        if self.split == 'test':
            random.setstate(state)

        return dict_return

    def __len__(self):
        if self.select_indices is not None:
            return len(self.select_indices)
        elif self.proportion_samples is not None:
            assert 0 < self.proportion_samples <= 1.
            return int(len(self.clip_ids) * self.proportion_samples)
        else:
            return len(self.clip_ids)

    def collate_fn(self, batch):
        """Overwrite for the PCA experiments"""
        if self.pca_augmentation is not None:
            assert self.pca_augmentation in ['speed', 'temporal_offset', 'spatial_flip']
            """Augment batch with similar segments for MDS experiment"""
            segment_inputs = ['past', 'future', 'all']
            time_inputs = ['time_indices_past', 'time_indices_future', 'time_indices_all']
            num_repetitions = 2 if self.pca_augmentation == 'spatial_flip' else 30  # We need a good sample size
            batch_new = []
            for i, sample in enumerate(batch):
                for j in range(num_repetitions):
                    new_sample = sample.copy()
                    for inp in segment_inputs:
                        if inp in sample:
                            new_sample[inp] = self.augment_trajectories(sample[inp], j)
                    for inp in time_inputs:
                        if inp in sample:
                            new_sample[inp] = self.augment_time_inputs(sample[inp], j, num_repetitions)
                    new_sample['index'] = torch.stack([torch.tensor(sample['index']), torch.tensor(j)])
                    batch_new.append(new_sample)
            batch = batch_new
        return torch.utils.data._utils.collate.default_collate(batch)


class FineGym(HumanKeypointsDataset):
    def __init__(self, split, **kwargs):
        self.iou_min = 0.1
        super().__init__(split, dataset_name='finegym', **kwargs)

        # Load label information, for inference-time tests
        with open(self.dataset_path / 'annotations' / 'finegym_annotation_info_v1.1.json', 'r') as f:
            video_info = json.load(f)
        dict_labels = {}
        for youtube_id, info in video_info.items():
            for event_id, v in info.items():
                dict_labels[youtube_id + '_' + event_id] = v['event']

        self.dict_labels = dict_labels

        self.label_categories = {
            1: 'Vault W',
            2: 'Floor Exercise W',
            3: 'Balance Beam W',
            4: 'Uneven Bars W',
            5: 'Vault M',
            6: 'Floor Exercise M',
            7: 'Pommel Horse M',
            8: 'Still Rings M',
            9: 'Parallel Bars M',
            10: 'High Bar M'
        }

        # Visualize
        # if split == 'test':
        #     to_visualize = 10
        #     clip_indices = range(len(self.clip_ids))
        #     clip_indices = [clip_indices[i] for i in range(to_visualize)]
        #     for clip_index in clip_indices:
        #         self.visualize_sample(clip_index)

    def get_video_path(self, event_id):
        """
        Returns the path of the video associated to an event ID
        :return:
        """
        return self.dataset_path / 'event_videos' / f'{event_id}.mp4'

    def get_label(self, index):
        """
        Returns the label associated to an index
        """
        event_id, *_ = self.clip_infos[self.clip_ids[index]]
        label = self.label_categories[self.dict_labels[event_id]]

        return label


class Diving48(HumanKeypointsDataset):
    def __init__(self, split, **kwargs):
        self.iou_min = 0.1
        super().__init__(split, dataset_name='diving48', **kwargs) / p

        # Visualize
        # if split == 'test':
        #     to_visualize = 10
        #     for clip_index in range(len(self.clip_ids)):
        #         self.visualize_sample(clip_index)
        #         if clip_index >= to_visualize:
        #             break

    def get_video_path(self, event_id):
        return self.dataset_path / 'rgb' / f'{event_id}.mp4'


class FisV(HumanKeypointsDataset):
    def __init__(self, split, **kwargs):
        self.iou_min = 0.1
        super().__init__(split, dataset_name='fisv', **kwargs)

        # Visualize
        # if split == 'test':
        #     to_visualize = 10
        #     clip_indices = range(len(self.clip_ids))
        #     for clip_index in clip_indices:
        #         self.visualize_sample(clip_index)
        #         if clip_index >= to_visualize:
        #             break

    def get_video_path(self, event_id):
        return self.dataset_path / 'videos' / f'{event_id}.mp4'
