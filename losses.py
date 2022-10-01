"""
Collection of metrics and losses to evaluate the performance of the model
"""

import os
import random
from dataclasses import dataclass
from typing import Dict, Tuple
from typing import Union

import numpy as np
import torch
import torch.distributed
import torchmetrics
from box_embeddings.common.utils import log1mexp

import distances
from utils.utils import my_autocast
from utils.utils import random_derangement, compress_tensor

"""
Values of the negative_pairs dict

For symmetric, and output of get_negative_pair() for both symmetric and asymmetric
- 1: positives
- 2: soft positive
- 3: soft negative
- 4: hard negative
- 5: itself

For asymmetric:
- 1: positives. P(A|B) = P(B|A) = 1
- 2: A contains B. P(A|B) = 1, 0 < P(B|A) < 1
    [note: this is in latent space, which means that in the trajectory space, the segment B contains A]
- 3: B contains A. P(B|A) = 1, 0 < P(A|B) < 1
- 4. overlap. 0 < P(B|A) < 1, 0 < P(A|B) < 1
- 5: (hard) negatives. P(A|B) = P(B|A) = 0
- 6: itself [no loss applied here]

This assumes tra_dec_p_inp_p and tra_dec_f_inp_p (or tra_dec_f_inp_f and tra_dec_p_inp_f) come from the same
sample from the distribution in latent space, but are evaluated at different times. If they come from
different samples, they are probably negatives, not soft positives.

frozenset makes the order of the elements in the key irrelevant

The first term is the value assuming a distance (symmetric). The second value assumes P(A|B), so asymmetric.

The elements marked with a * are hard positives that would make more sense to have as soft positives, but they are hard
so that in the triplet loss some hard negatives associated to them can be used as hard negatives. This only applies in 
the symmetrical case
"""

negative_pairs = {
    ('tra_inp_p', 'tra_inp_f'): (11, 4),
    ('tra_inp_p', 'tra_dec_p_inp_p'): (1, 1),
    ('tra_inp_p', 'tra_dec_f_inp_f'): (2, 4),
    ('tra_inp_p', 'tra_dec_p_inp_f'): (4, 5),
    ('tra_inp_p', 'tra_dec_f_inp_p'): (1, 4),  # *
    ('tra_inp_p', 'tra_dec_a_inp_a'): (2, 2),
    ('tra_inp_p', 'tra_dec_p_inp_p__inter__inp_f'): (2, 1),
    ('tra_inp_p', 'tra_dec_f_inp_p__inter__inp_f'): (2, 4),
    ('tra_inp_p', 'tra_inp_a'): (11, 2),
    ('tra_inp_p', 'tra_inp_p__inter__inp_f'): (5, 6),
    ('tra_inp_p', 'tra_dec_a_inp_p'): (2, 2),
    ('tra_inp_p', 'tra_dec_a_inp_f'): (3, 5),
    ('tra_inp_p', 'tra_dec_p_inp_a'): (1, 1),
    ('tra_inp_p', 'tra_dec_f_inp_a'): (2, 4),

    ('tra_inp_f', 'tra_dec_p_inp_p'): (2, 4),
    ('tra_inp_f', 'tra_dec_f_inp_f'): (1, 1),
    ('tra_inp_f', 'tra_dec_p_inp_f'): (1, 4),  # *
    ('tra_inp_f', 'tra_dec_f_inp_p'): (4, 5),
    ('tra_inp_f', 'tra_dec_a_inp_a'): (2, 2),
    ('tra_inp_f', 'tra_dec_p_inp_p__inter__inp_f'): (2, 4),
    ('tra_inp_f', 'tra_dec_f_inp_p__inter__inp_f'): (2, 1),
    ('tra_inp_f', 'tra_inp_a'): (11, 2),
    ('tra_inp_f', 'tra_inp_p__inter__inp_f'): (5, 6),
    ('tra_inp_f', 'tra_dec_a_inp_p'): (3, 5),
    ('tra_inp_f', 'tra_dec_a_inp_f'): (2, 2),
    ('tra_inp_f', 'tra_dec_p_inp_a'): (2, 4),
    ('tra_inp_f', 'tra_dec_f_inp_a'): (1, 1),

    ('tra_dec_p_inp_p', 'tra_dec_f_inp_f'): (2, 4),
    ('tra_dec_p_inp_p', 'tra_dec_p_inp_f'): (3, 5),
    ('tra_dec_p_inp_p', 'tra_dec_f_inp_p'): (1, 4),  # *
    ('tra_dec_p_inp_p', 'tra_dec_a_inp_a'): (2, 2),
    ('tra_dec_p_inp_p', 'tra_dec_p_inp_p__inter__inp_f'): (2, 1),
    ('tra_dec_p_inp_p', 'tra_dec_f_inp_p__inter__inp_f'): (2, 4),
    ('tra_dec_p_inp_p', 'tra_inp_a'): (2, 2),
    ('tra_dec_p_inp_p', 'tra_inp_p__inter__inp_f'): (2, 2),
    ('tra_dec_p_inp_p', 'tra_dec_a_inp_p'): (2, 2),
    ('tra_dec_p_inp_p', 'tra_dec_a_inp_f'): (3, 5),
    ('tra_dec_p_inp_p', 'tra_dec_p_inp_a'): (2, 1),
    ('tra_dec_p_inp_p', 'tra_dec_f_inp_a'): (2, 4),

    ('tra_dec_f_inp_f', 'tra_dec_p_inp_f'): (1, 4),  # *
    ('tra_dec_f_inp_f', 'tra_dec_f_inp_p'): (3, 5),
    ('tra_dec_f_inp_f', 'tra_dec_a_inp_a'): (2, 2),
    ('tra_dec_f_inp_f', 'tra_dec_p_inp_p__inter__inp_f'): (2, 4),
    ('tra_dec_f_inp_f', 'tra_dec_f_inp_p__inter__inp_f'): (2, 1),
    ('tra_dec_f_inp_f', 'tra_inp_a'): (2, 2),
    ('tra_dec_f_inp_f', 'tra_inp_p__inter__inp_f'): (2, 2),
    ('tra_dec_f_inp_f', 'tra_dec_a_inp_p'): (3, 5),
    ('tra_dec_f_inp_f', 'tra_dec_a_inp_f'): (2, 2),
    ('tra_dec_f_inp_f', 'tra_dec_p_inp_a'): (2, 4),
    ('tra_dec_f_inp_f', 'tra_dec_f_inp_a'): (2, 1),

    ('tra_dec_p_inp_f', 'tra_dec_f_inp_p'): (3, 5),
    ('tra_dec_p_inp_f', 'tra_dec_a_inp_a'): (3, 5),
    ('tra_dec_p_inp_f', 'tra_dec_p_inp_p__inter__inp_f'): (3, 5),
    ('tra_dec_p_inp_f', 'tra_dec_f_inp_p__inter__inp_f'): (2, 4),
    ('tra_dec_p_inp_f', 'tra_inp_a'): (4, 5),
    ('tra_dec_p_inp_f', 'tra_inp_p__inter__inp_f'): (4, 5),
    ('tra_dec_p_inp_f', 'tra_dec_a_inp_p'): (3, 5),
    ('tra_dec_p_inp_f', 'tra_dec_a_inp_f'): (3, 5),
    ('tra_dec_p_inp_f', 'tra_dec_p_inp_a'): (3, 5),
    ('tra_dec_p_inp_f', 'tra_dec_f_inp_a'): (2, 4),

    ('tra_dec_f_inp_p', 'tra_dec_a_inp_a'): (3, 5),
    ('tra_dec_f_inp_p', 'tra_dec_p_inp_p__inter__inp_f'): (2, 4),
    ('tra_dec_f_inp_p', 'tra_dec_f_inp_p__inter__inp_f'): (3, 5),
    ('tra_dec_f_inp_p', 'tra_inp_a'): (4, 5),
    ('tra_dec_f_inp_p', 'tra_inp_p__inter__inp_f'): (4, 5),
    ('tra_dec_f_inp_p', 'tra_dec_a_inp_p'): (3, 5),
    ('tra_dec_f_inp_p', 'tra_dec_a_inp_f'): (3, 5),
    ('tra_dec_f_inp_p', 'tra_dec_p_inp_a'): (2, 4),
    ('tra_dec_f_inp_p', 'tra_dec_f_inp_a'): (3, 5),

    ('tra_dec_a_inp_a', 'tra_dec_p_inp_p__inter__inp_f'): (2, 3),
    ('tra_dec_a_inp_a', 'tra_dec_f_inp_p__inter__inp_f'): (2, 3),
    ('tra_dec_a_inp_a', 'tra_inp_a'): (1, 1),
    ('tra_dec_a_inp_a', 'tra_inp_p__inter__inp_f'): (2, 1),
    ('tra_dec_a_inp_a', 'tra_dec_a_inp_p'): (3, 5),
    ('tra_dec_a_inp_a', 'tra_dec_a_inp_f'): (3, 5),
    ('tra_dec_a_inp_a', 'tra_dec_p_inp_a'): (2, 3),
    ('tra_dec_a_inp_a', 'tra_dec_f_inp_a'): (2, 3),

    ('tra_dec_p_inp_p__inter__inp_f', 'tra_dec_f_inp_p__inter__inp_f'): (2, 4),
    ('tra_dec_p_inp_p__inter__inp_f', 'tra_inp_a'): (2, 2),
    ('tra_dec_p_inp_p__inter__inp_f', 'tra_inp_p__inter__inp_f'): (1, 2),
    ('tra_dec_p_inp_p__inter__inp_f', 'tra_dec_a_inp_p'): (2, 2),
    ('tra_dec_p_inp_p__inter__inp_f', 'tra_dec_a_inp_f'): (3, 5),
    ('tra_dec_p_inp_p__inter__inp_f', 'tra_dec_p_inp_a'): (2, 1),
    ('tra_dec_p_inp_p__inter__inp_f', 'tra_dec_f_inp_a'): (2, 4),

    ('tra_dec_f_inp_p__inter__inp_f', 'tra_inp_a'): (2, 2),
    ('tra_dec_f_inp_p__inter__inp_f', 'tra_inp_p__inter__inp_f'): (1, 2),
    ('tra_dec_f_inp_p__inter__inp_f', 'tra_dec_a_inp_p'): (3, 5),
    ('tra_dec_f_inp_p__inter__inp_f', 'tra_dec_a_inp_f'): (2, 2),
    ('tra_dec_f_inp_p__inter__inp_f', 'tra_dec_p_inp_a'): (2, 4),
    ('tra_dec_f_inp_p__inter__inp_f', 'tra_dec_f_inp_a'): (2, 1),

    ('tra_inp_a', 'tra_inp_p__inter__inp_f'): (11, 1),
    ('tra_inp_a', 'tra_dec_a_inp_p'): (3, 5),
    ('tra_inp_a', 'tra_dec_a_inp_f'): (3, 5),
    ('tra_inp_a', 'tra_dec_p_inp_a'): (2, 3),
    ('tra_inp_a', 'tra_dec_f_inp_a'): (2, 3),

    ('tra_inp_p__inter__inp_f', 'tra_dec_a_inp_p'): (3, 5),
    ('tra_inp_p__inter__inp_f', 'tra_dec_a_inp_f'): (3, 5),
    ('tra_inp_p__inter__inp_f', 'tra_dec_p_inp_a'): (2, 3),
    ('tra_inp_p__inter__inp_f', 'tra_dec_f_inp_a'): (2, 3),

    ('tra_dec_a_inp_p', 'tra_dec_a_inp_f'): (3, 5),
    ('tra_dec_a_inp_p', 'tra_dec_p_inp_a'): (2, 3),
    ('tra_dec_a_inp_p', 'tra_dec_f_inp_a'): (3, 5),

    ('tra_dec_a_inp_f', 'tra_dec_p_inp_a'): (3, 5),
    ('tra_dec_a_inp_f', 'tra_dec_f_inp_a'): (2, 3),

    ('tra_dec_p_inp_a', 'tra_dec_f_inp_a'): (1, 4),

    # This compares two different reencoded samples (sampled before decoding) from the same distribution
    ('tra_dec_p_inp_p', 'tra_dec_p_inp_p'): (1, 1),
    ('tra_dec_f_inp_f', 'tra_dec_f_inp_f'): (1, 1),
    ('tra_dec_p_inp_f', 'tra_dec_p_inp_f'): (4, 5),
    ('tra_dec_f_inp_p', 'tra_dec_f_inp_p'): (4, 5),
    ('tra_dec_a_inp_a', 'tra_dec_a_inp_a'): (1, 1),
    ('tra_dec_p_inp_p__inter__inp_f', 'tra_dec_p_inp_p__inter__inp_f'): (1, 1),
    ('tra_dec_f_inp_p__inter__inp_f', 'tra_dec_f_inp_p__inter__inp_f'): (1, 1),
    ('tra_dec_a_inp_p', 'tra_dec_a_inp_p'): (3, 5),
    ('tra_dec_a_inp_f', 'tra_dec_a_inp_f'): (3, 5),
    ('tra_dec_p_inp_a', 'tra_dec_p_inp_a'): (1, 1),
    ('tra_dec_f_inp_a', 'tra_dec_f_inp_a'): (1, 1),
}


def get_negative_pair(a, b, symmetric=True, all_hard_positives=False, all_hard_negatives=False):
    """
    Return relationship between a and b
    If not symmetric, this is the legend for P(B|A):
        1. P(B|A) = 1 (positive)
        2. 0 < P(B|A) < 1  (soft positive)
        4. P(B|A) = 0  (hard negative)
        5. itself

    all_hard_negatives and all_hard_positives modify all the "soft" (negatives or positives) to make them hard.
    """
    if '_s_' in a:  # '_s_' indicates the start of the sample id, but the actual name comes before
        a = a.split('_s_')[0]
    if '_s_' in b:
        b = b.split('_s_')[0]
    key, reverse = ((a, b), False) if (a, b) in negative_pairs else ((b, a), True)
    if symmetric:
        negative_pair = negative_pairs[key][0]
    else:
        value = negative_pairs[key][1]
        if value == 1:
            p_ba = 1
        elif value == 2:
            p_ba = 1 if reverse else 2
        elif value == 3:
            p_ba = 2 if reverse else 1
        elif value == 4:
            p_ba = 2
        elif value == 5:
            p_ba = 4
        else:  # value = 6
            p_ba = 5
        negative_pair = p_ba

    if all_hard_positives and negative_pair == 2:
        negative_pair = 1

    elif all_hard_negatives and negative_pair == 3:
        negative_pair = 4

    return negative_pair


@dataclass(eq=False)
class TrajectoryLoss:
    """As a class so that we can have attributes and pre-compute parameters"""
    all_hard_positives: bool = False
    all_hard_negatives: bool = False

    def __call__(self, generate_extrapolation=False, reencode=False, use_all=False, reconstruct_intersection=False,
                 symmetric_dist=True, **kwargs):
        list_tensors = ['tra_inp_p', 'tra_inp_f']
        if use_all:
            list_tensors += ['tra_inp_a']
        if reconstruct_intersection:
            list_tensors += ['tra_inp_p__inter__inp_f']
        if reencode:
            list_tensors += ['tra_dec_p_inp_p', 'tra_dec_f_inp_f']
            if generate_extrapolation:
                list_tensors += ['tra_dec_p_inp_f', 'tra_dec_f_inp_p']
            if use_all:
                list_tensors += ['tra_dec_a_inp_a']
                if symmetric_dist:  # See comment in Trainer
                    list_tensors += ['tra_dec_a_inp_p', 'tra_dec_a_inp_f', 'tra_dec_p_inp_a', 'tra_dec_f_inp_a']
            if reconstruct_intersection:
                list_tensors += ['tra_dec_p_inp_p__inter__inp_f', 'tra_dec_f_inp_p__inter__inp_f']

        list_tensors_expanded = []  # In case option_reencode == 2
        trajs = []
        for name in list_tensors:
            traj = kwargs[name]['tensor']
            if len(traj.shape) == 3:  # Multiple samples
                for i in range(traj.shape[1]):  # Number of samples
                    list_tensors_expanded.append(name + f'_s_{i}')
                    trajs.append(traj[:, i])
            else:
                list_tensors_expanded.append(name)
                trajs.append(traj)

        trajs = torch.stack(trajs, dim=1)
        list_tensors = list_tensors_expanded

        negative_matrix = 5 * torch.ones((len(list_tensors), len(list_tensors))).long()
        for i, name_i in enumerate(list_tensors):
            for j, name_j in enumerate(list_tensors):
                if i != j:
                    negative_matrix[i, j] = get_negative_pair(name_i, name_j, symmetric=symmetric_dist,
                                                              all_hard_positives=self.all_hard_positives,
                                                              all_hard_negatives=self.all_hard_negatives)

        return trajectory_loss_(trajs, negative_matrix, symmetric_dist=symmetric_dist, **kwargs)


def trajectory_loss_(trajs: torch.tensor, negative_matrix, distance_type='kl-divergence', loss_type='contrastive',
                     latent_distribution='gaussian', margin=1., τ=0.1, num_latent_params=2, values=None,
                     symmetric_dist=True, **kwargs):
    """

    :param trajs: [B, M, N*num_latent_params]
    :param negative_matrix: [M, M] matrix, where every tensor pair is given an index:
        - 1: positives
        - 2: soft positive
        - 3: soft negative
        - 4: hard negative
        - 5: itself
        Other than that, all the other B-1 elements in a tensor are considered regular negatives
    :param distance_type:
    :param loss_type: ['contrastive', 'triplet', 'bce']
    :param margin:
    :param τ:
    :param num_latent_params:
    :param latent_distribution:
    :param values: used to weigh positive and negative losses
    :param symmetric_dist: if True, the loss assumes a symmetric distance, where the distance is a proper metric. If
        False, the loss assumes a containment setting
    :param kwargs:
    :return:
    """
    assert loss_type in ['contrastive', 'triplet', 'bce']
    if not symmetric_dist:
        assert distance_type in ['kl-divergence', 'prediction']

    dist_fn = distances.get_dist_fn(latent_distribution)

    param_1, param_2 = distances.get_params(trajs, num_latent_params)

    to_report = {  # In the case of box_embeddings this may not be the mean. It can be the first coordinate of the box
        'mean_traj_norm': ('histogram', param_1.pow(2).sum(-1).sqrt().view(-1)),
    }
    if param_2 is not None:
        to_report['var_traj_norm'] = ('histogram', param_2.pow(2).sum(-1).sqrt().view(-1))

    if loss_type in ['triplet', 'bce']:
        # We use same strategy to decide positives and negatives. In BCE we do not need a specific negative for every
        # specific positive, but it is convenient because it is good to have a balanced distribution (50/50)
        loss_fn = {'triplet': triplet, 'bce': bce}[loss_type]

        pairs_positive_negative, set_compute = get_pairs_positive_negative(param_1.shape[0], negative_matrix)
        if len(pairs_positive_negative) == 0:
            loss = torch.zeros(1).to(param_1.device)
        else:
            # First, precompute pairs
            idx = torch.tensor(list(set_compute)).to(param_1.device)
            v = dist_fn(torch.index_select(param_1, 1, idx[:, 0]), torch.index_select(param_1, 1, idx[:, 1]),
                        torch.index_select(param_2, 1, idx[:, 0]), torch.index_select(param_2, 1, idx[:, 1]),
                        distance_type=distance_type)
            precomputed_results = {k: v[:, i] for i, k in enumerate(set_compute)}

            # Now compute losses
            pos_distances_report = []
            pos_negatives_report = []

            losses = []
            for pos, negs, w in pairs_positive_negative:
                pos_distance = precomputed_results[pos]
                for neg in negs:
                    query_idx, target_idx = neg
                    if type(target_idx) == int:
                        neg_distance = precomputed_results[(query_idx, target_idx)]
                    else:
                        param_2_pred = param_2[:, query_idx] if param_2 is not None else None
                        param_2_gt = param_2[target_idx[:, 0], target_idx[:, 1]] if param_2 is not None else None
                        neg_distance = dist_fn(param_1[:, query_idx], param_1[target_idx[:, 0], target_idx[:, 1]],
                                               param_2_pred, param_2_gt, distance_type=distance_type)

                    loss = loss_fn(pos_distance, neg_distance, margin)
                    loss = loss.mean()  # Mean across elements in the batch
                    losses.append(w * loss)

                    pos_negatives_report.append(neg_distance)
                pos_distances_report.append(pos_distance)

            loss = torch.stack(losses).mean()
            to_report['positive_dist'] = ('value', torch.cat(pos_distances_report, dim=0).mean())
            to_report['negative_dist'] = ('value', torch.cat(pos_negatives_report, dim=0).mean())

    else:  # loss_type == 'contrastive':
        # This results in too large of a batch size
        # if torch.distributed.is_available() and torch.distributed.is_initialized() and loss_type == 'contrastive':
        #     param_1 = SyncFunction.apply(param_1.contiguous())  # [B*#gpus, M, N]
        #     param_2 = SyncFunction.apply(param_2.contiguous()) if param_2 is not None else None  # [B*#gpus, M, N]

        if values is None:
            # hard positives: +1, soft positives: +0.2, soft negatives: 0, hard negatives: 0, self: 0
            # Having negative values (for hard negatives) results in model just focusing on that and making loss -inf
            # So instead, we weight them later accordingly in the denominator
            values = [0, 1, 0.2, 0, 0, 0]  # The first zero is not used
        w_hard_negs = 20  # x20 the importance of a regular negative
        w_soft_negs = 5

        # Combine B and M
        batch_size = param_1.shape[0]
        param_1 = param_1.view(-1, param_1.shape[-1])
        param_1_query = param_1[:, None]
        param_1_target = param_1[None, :]
        if param_2 is not None:
            param_2 = param_2.view(-1, param_1.shape[-1])
            param_2_query = param_2[:, None].float()
            param_2_target = param_2[None, :].float()
        else:
            param_2_query, param_2_target = None, None

        dist_matrix = dist_fn(param_1_query, param_1_target, param_2_query, param_2_target,
                              distance_type=distance_type).type(param_1_target.type())

        negative_matrix_expanded = torch.block_diag(*([negative_matrix] * batch_size))
        mask = (negative_matrix_expanded == 5).long().to(dist_matrix.device)
        # mask = mask.to(dist_matrix.device)
        dist_matrix = dist_matrix * (1 - mask) + torch.tensor(1e12) * mask

        neg_weight_matrix = torch.zeros_like(dist_matrix, device='cpu')
        neg_weight_matrix[negative_matrix_expanded == 3] = torch.log(torch.tensor(w_soft_negs))
        neg_weight_matrix[negative_matrix_expanded == 4] = torch.log(torch.tensor(w_hard_negs))
        dist_matrix += neg_weight_matrix.to(dist_matrix.device)

        score = -dist_matrix / τ
        score = torch.log_softmax(score, dim=-1)

        values = torch.tensor(values)

        gt_values = values[negative_matrix_expanded % 10].to(score.device)
        assert gt_values.sum() > 0  # Make sure the positives are more important than negatives!
        loss = - score * gt_values  # Negative because it is the negative log-likelihood

        score_ = score.detach()
        to_report['score_hard_positives'] = ('value', score_[negative_matrix_expanded == 1].exp().mean())
        to_report['score_soft_positives'] = ('value', score_[negative_matrix_expanded == 2].exp().mean())
        to_report['score_hard_negatives'] = ('value', score_[negative_matrix_expanded == 4].exp().mean())
        to_report['score_regular_negatives'] = ('value', score_[negative_matrix_expanded == 0].exp().mean())

        loss = loss.sum(-1).mean()  # This is doing the sum across B*M, and mean across the other B*M

    for_metrics = {}
    return loss, to_report, for_metrics


def triplet(pos_distance, neg_distance, margin: Union[str, float] = 1.0):
    if type(margin) == str:
        # should be of the form 'percentile_X', where X is a float from 0 to 1
        # A percentile of 1.0 will have the loss apply to all pos/neg pairs. A percentile 0.0, to none.
        assert margin.startswith('percentile')
        margin = np.percentile((neg_distance - pos_distance).cpu().detach().numpy(), float(margin.split('_')[-1]) * 100)
    loss = torch.maximum(pos_distance - neg_distance + margin, torch.tensor(0))
    return loss


@my_autocast
def bce(pos_distance, neg_distance, *args):
    """
    Binary Cross Entropy. Assumes the distance is -log_prob. For example -log(prob(a|b)). In practice, it will work
    for any distance that is in [0, inf), so that log_prob is in (-inf, 0]
    """
    log_prob_pos = - pos_distance.float()
    log_prob_neg = - neg_distance.float()
    loss = - (log_prob_pos + log1mexp(log_prob_neg.float()))

    return loss


def get_pairs_positive_negative(batch_size, negatives_matrix, non_sym_distance=False, p_soft_pos=0, p_soft_neg=None):
    """
    Returns pairs of positive/negative.
    It returns a set of indexes of pairs of tensors to be computed. And it also returns (negative, positive) pairs of
    tensor pairs. This is returned separately in case some tensor pairs are repeated in the (negative, positive) pairs
    list. No need to compute the distance twice.
    For every positive index in the upper diagonal, and for every element b in that index, this returns two negative
    indices, one sampled from the row and the other sampled from the column of that index. The candidates to be sampled
    are all the other B-1 elements (from any tensor), as well as the elements from b that are marked as soft negative.
    Then, another set of negatives is returned from each hard negative index in the column and row corresponding to the
    index of the positive.
    Soft positives are used the same way as hard positives, but only with probability p_soft_pos
    :param batch_size: B
    :param negatives_matrix:
    :param non_sym_distance: distance metric is not symmetric. Compute both ways.
    :param p_soft_pos: Probability that we create positives out of soft positives
    :param p_soft_neg: Probability that soft negatives appear instead of an element of another sample from the batch. If
        None, the probability is 1/B
    :return: index_set, pairs
    """

    max_sample = np.minimum(5, negatives_matrix.shape[0])

    # The random_derangement is significantly more expensive than a random permutation, so we pre-compute a few
    # num_random_derangements = 100
    # random_derangements = torch.tensor(np.array(
    #     [random_derangement(batch_size) for _ in range(num_random_derangements)]))

    # To make it even faster, we just do a circular permutation
    if batch_size == 1:  # Cannot compute negatives with only 1 element
        return [], set()

    random_derangements = torch.stack([torch.roll(torch.arange(batch_size), shifts=(i,)) for i in range(1, batch_size)])
    num_random_derangements = batch_size - 1

    matrix_upper = torch.triu(negatives_matrix, diagonal=1)  # Force symmetrical (it should already be)

    if non_sym_distance:
        positives = negatives_matrix % 10 == 1
        soft_positives = negatives_matrix == 2
    else:
        # For the negatives we still look at the whole negatives_matrix because row != column
        positives = matrix_upper % 10 == 1
        soft_positives = matrix_upper == 2

    # Even if symmetric, the positives will be the same but the negatives will be different. So need to consider the two
    # directions.
    # positives = matrix_ == 1
    # soft_positives = matrix_ == 2

    positives = positives + soft_positives * (torch.rand(soft_positives.shape) < p_soft_pos)
    positives = torch.where(positives)

    set_compute = set()
    negatives_all = []
    for row, col in zip(*positives):
        w = 1 if negatives_matrix[row, col] == 1 else 10  # Weigh more the pairs that contain a 11
        set_compute.add((int(row), int(col)))
        negatives = []

        # For every positive element in the matrix, we create 1 set of "regular" negatives in the column, another set of
        # "regular" negatives in the row, and then one set of negatives for every hard negative (both columns and rows).

        # First compute regular negatives
        for direction in ['rows', 'columns']:  # 'rows' means that we navigate through rows, fixing the column to col
            candidates = []
            # for i in range(negatives_matrix.shape[0]):
            # We do not consider all options because it is a bit slow, and they are regular negatives anyway
            for i in np.random.choice(negatives_matrix.shape[0], max_sample, replace=False):
                val = negatives_matrix[i, col] if direction == 'rows' else negatives_matrix[row, i]
                val = val % 10
                if val in [0, 1, 2]:  # Cannot sample positive as negative
                    # cand = torch.tensor(random_derangement(batch_size))
                    cand = random_derangements[random.randint(0, num_random_derangements - 1)]
                else:  # negatives_matrix[., .] in [3, 4]
                    if p_soft_neg is None:
                        cand = torch.randperm(batch_size)
                    else:
                        cand_same = torch.arange(batch_size)
                        # cand_diff = torch.tensor(random_derangement(batch_size))
                        cand_diff = random_derangements[random.randint(0, num_random_derangements - 1)]
                        mask_same = torch.rand(batch_size) < p_soft_neg
                        cand = cand_same * mask_same + cand_diff * ~mask_same
                cand = torch.stack([cand, i * torch.ones_like(cand), ], dim=-1)
                candidates.append(cand)
            candidates = torch.stack(candidates, dim=0)
            indices = torch.randint(0, candidates.shape[0], (batch_size,))
            negatives_direction = candidates[indices, torch.arange(batch_size)]
            negatives.append((int(col) if direction == 'rows' else int(row), negatives_direction))

        # Next we prepare hard negatives
        for neg_col in torch.where(negatives_matrix[row] == 4)[0]:
            negatives.append((int(row), int(neg_col)))
            set_compute.add((int(row), int(neg_col)))

        for neg_row in torch.where(negatives_matrix[:, col] == 4)[0]:
            negatives.append((int(neg_row), int(col)))
            set_compute.add((int(neg_row), int(col)))

        positives = (int(row), int(col))
        negatives_all.append((positives, negatives, w))

    return negatives_all, set_compute


def reconstruction_loss_traj(seg_inp_p, seg_inp_f, seg_dec_p_inp_p, seg_dec_f_inp_f, seg_dec_f_inp_p=None,
                             seg_dec_p_inp_f=None, seg_dec_p_inp_p__inter__inp_f=None,
                             seg_dec_f_inp_p__inter__inp_f=None, distance_fn_name='euclidean_l2',
                             loss_type='regression', margin=0.5, generate_extrapolation=False,
                             reconstruct_intersection=None, temporal_negs=False, param_dist=None, **kwargs):
    """
    Computes the distance (point to point) in the input space
    :param seg_inp_p: ground truth input points, past
    :param seg_inp_f: ground truth input points, future
    :param seg_dec_p_inp_p: predicted input points, past from the past
    :param seg_dec_f_inp_f: predicted input points, future from the future
    :param seg_dec_f_inp_p: predicted input points, future from the past
    :param seg_dec_p_inp_f: predicted input points, past from the future
    :param seg_dec_p_inp_p__inter__inp_f: predicted input points, past from intersection
    :param seg_dec_f_inp_p__inter__inp_f: predicted input points, future from intersection
    :param distance_fn_name:
    :param loss_type: ['regression', 'contrastive', 'triplet'], or a list with more than one
    :param τ: temperature, in case loss_type == 'contrastive'
    :param margin: margin for triplet loss
    :param generate_extrapolation: compute loss on points that are not in the input, but in the associated view
    :param reconstruct_intersection:
    :param temporal_negs: int / bool if loss_type is triplet or contrastive, we can create hard negs from temporal negs
    :param param_dist: Extra information to define the distance function
    :param kwargs:
    :return:
    """

    query = torch.cat([seg_dec_p_inp_p['tensor'], seg_dec_f_inp_f['tensor']], dim=0)
    target = torch.cat([seg_inp_p['tensor'], seg_inp_f['tensor']], dim=0)  # ground truth
    duration = torch.cat([seg_inp_p['seg_len'], seg_inp_f['seg_len']], dim=0)

    if reconstruct_intersection:
        assert seg_dec_p_inp_p__inter__inp_f is not None and seg_dec_f_inp_p__inter__inp_f is not None
        query = torch.cat([query, seg_dec_p_inp_p__inter__inp_f['tensor'],
                           seg_dec_f_inp_p__inter__inp_f['tensor']], dim=0)
        target = torch.cat([target, seg_inp_p['tensor'], seg_inp_f['tensor']], dim=0)
        duration = torch.cat([duration, seg_inp_p['seg_len'], seg_inp_f['seg_len']], dim=0)

    # [Comment for contrastive loss case] We do not share across GPUs because this results in (B x T x S x B x T).
    # Without sharing, we still have a lot of comparisons but not as many (B/#GPUs x T x S x B/#GPUs x T).

    # "Compressed" basically means that the temporal dimension is absorbed by the batch dimension
    query = compress_tensor(query, duration)
    target = compress_tensor(target, duration)

    return reconstruction_loss(query, target, duration, loss_type, distance_fn_name, margin, temporal_negs,
                               param_dist=param_dist)


def reconstruction_loss_categorical(latent_distribution, z_distr_reconstruct, query, target, **kwargs):
    """
    For the Trajectron++ baseline, where samples coming from different categorical options are provided, and each
    option has an associated weight, that weighs the loss.
    """
    loss, to_report, for_metrics = reconstruction_loss(query, target, **kwargs, average=False)
    if latent_distribution == 'categorical':
        z_distr = categorical_softmax(z_distr_reconstruct, dim=-1)
        # If loss only contains one sample (instead of all of them), the following sum will just be equal to loss.mean()
        if z_distr.shape[-1] == loss.shape[-1]:
            loss = (loss * z_distr).sum(-1).mean()
        else:
            # We do not sample all possibilities. Inference time, so loss is just for reference
            loss = loss.mean()  # We could also do some max() over samples. Not important
        return loss, to_report, for_metrics
    else:
        return torch.mean(loss), to_report, for_metrics


def reconstruction_loss_mtp(seg_inp_p, seg_dec_f_inp_p, **kwargs):
    """
    For the MTP baseline. There are several heads in the decoder, and only the best prediction is taken into account in
    the loss. The loss is at the point (time-step) level, not whole-trajectory level.
    """
    query = seg_dec_f_inp_p['tensor']
    target = seg_inp_p['tensor']  # ground truth
    assert query.shape[-1] % target.shape[-1] == 0, 'Make sure M is an integer'
    m = query.shape[-1] // target.shape[-1]  # Number of predictions
    duration = seg_inp_p['seg_len']
    query = compress_tensor(query, duration)
    target = compress_tensor(target, duration)
    assert query.shape[1] == 1, 'Please only generate 1 sample (num_sample_points=1)'
    query = query.view(query.shape[0], m, -1)
    loss, to_report, for_metrics = reconstruction_loss(query, target, duration, **kwargs, average=False)
    loss = loss.min(-1)[0]  # Only take the minimum loss for every sample
    loss = loss.mean()
    return loss, to_report, for_metrics


def reconstruction_loss(query, target, duration=None, loss_type='regression', distance_fn_name='euclidean_l2',
                        margin=0.5, temporal_negs=False, average=True, param_dist=None, **kwargs) -> \
        Tuple[torch.tensor, Dict[str, torch.tensor], Dict[str, torch.tensor]]:
    distance_fn = distances.get_dist_fn(distance_fn_name, param_dist)

    to_report, for_metrics = {}, {}
    if type(loss_type) == str:
        loss_type = [loss_type]

    loss = 0
    if 'regression' in loss_type:
        distance_pairs = distance_fn(query, target, same_leading_dims=1)
        loss_regression = distance_pairs
        if average:
            loss_regression = loss_regression.mean()
            to_report['loss_regression'] = ('value', loss_regression)
        loss += loss_regression
    if 'triplet' in loss_type:
        """
        Instead of creating negatives at the trajectory level, we create negatives at the point level. Good because:
        1) mixes points from different temporal steps, 2) if some points are NaN, that comparison would be deleted,
        so long sequences would almost always have NaN negatives, and their last points never used in the loss
        We don't want different samples from same trajectory to be negatives b/w themselves (different temporal 
        steps yes). Right now, all S samples from one trajectory will be negatives with the S samples from the same 
        negative trajectory.
        """
        distance_positive = distance_fn(query, target, same_leading_dims=1)
        permutation = random_derangement(query.shape[0])
        distance_negative = distance_fn(query, target[permutation], same_leading_dims=1)
        loss_triplet = torch.maximum(distance_positive - distance_negative + margin, torch.tensor(0))
        if average:
            loss_triplet = loss_triplet.mean()
            to_report['distance_positive_mean'] = ('value', distance_positive.mean())
            to_report['distance_negative_mean'] = ('value', distance_negative.mean())
            to_report['loss_gen_triplet_normal'] = ('value', loss_triplet.detach())
        if temporal_negs:  # True or value != 0
            permutation_hard = []
            for d in duration.cpu():
                perm = random_derangement(d) if d > 1 else [0]
                permutation_hard += list(perm + len(permutation_hard))
            distance_negative_hard = distance_fn(query, target[permutation_hard], same_leading_dims=1)
            loss_hard = torch.maximum(distance_positive - distance_negative_hard + margin, torch.tensor(0))
            if average:
                loss_hard = loss_hard.mean()
                to_report['loss_gen_triplet_hard_temp'] = ('value', loss_hard)
            loss_triplet = (temporal_negs * loss_hard + loss_triplet)  # / (1 + temporal_negs)
        loss += loss_triplet
    if 'contrastive' in loss_type:
        raise NotImplementedError
    return loss, to_report, for_metrics


def kld_loss(z_distr, prior_distr, latent_distribution='gaussian', num_latent_params=2, **kwargs):
    assert latent_distribution in ['categorical', 'gaussian']
    if latent_distribution == 'categorical':
        assert num_latent_params == 1
        z_distr = torch.distributions.OneHotCategorical(categorical_softmax(z_distr, -1))
        prior_distr = torch.distributions.OneHotCategorical(categorical_softmax(prior_distr, -1))
        kl_value = torch.distributions.kl_divergence(z_distr, prior_distr)
        dist = kl_value
    else:  # 'gaussian'
        mean_z, logvar_z = distances.get_params(z_distr, num_latent_params)
        mean_prior, logvar_prior = distances.get_params(prior_distr, num_latent_params)
        dist = distances.gaussian(mean_prior, mean_z, logvar_a=logvar_prior, logvar_b=logvar_z,
                                  distance_type='kl-divergence')
        """
        # The previous code is equivalent to:
        distr_z = torch.distributions.MultivariateNormal(mean_z, torch.diag_embed(logvar_z.exp()))
        distr_prior = torch.distributions.MultivariateNormal(mean_prior, torch.diag_embed(logvar_prior.exp()))
        dist = torch.distributions.kl_divergence(distr_z, distr_prior)
        """
    loss = dist.mean()  # Mean across batch elements and across time
    to_report, for_metrics = {}, {}
    return loss, to_report, for_metrics


def info_loss(z_distr, prior_distr, latent_distribution='categorical', num_latent_params=2, info_prior=False, **kwargs):
    """
    Third term of the InfoVAE objective. Returns the negative of the mutual information
    For Categorical and Normal variables only.
    In theory, this computes Iq(prior, z), which is equivalent to  -KL(q(z)||p(z)), where p is the prior. In
    practice, following the InfoVAE paper, q(z) is approximated with 1/N*sum(q(z|x)), summing q(z|x) over the batch.

    info_prior: In Trajectron (where the prior depends on the batch element), they replace z_dist with prior_distr. If
        info_prior is True, we do the same
    """
    assert latent_distribution in ['categorical', 'gaussian']
    if info_prior:
        z_distr = prior_distr
    if latent_distribution == 'categorical':
        """The mixture of a categorical distribution is a categorical distribution, with probabilities for every element
        being the average of probabilities of the distributions"""
        prior_distr = categorical_softmax(prior_distr, dim=-1)
        z_distr = categorical_softmax(z_distr, dim=-1)
        dist_p = torch.distributions.OneHotCategorical(probs=prior_distr)
        dist_q = torch.distributions.OneHotCategorical(probs=z_distr.mean(dim=0))
    else:  # gaussian
        # Note that the entropy of a GMM is not implemented in PyTorch. We just leave this here to build on top of if
        # ever used.
        mean, logvar = distances.get_params(prior_distr, num_latent_params)
        dist_p = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=torch.diag_embed(logvar.exp()))
        dist_qx = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=torch.diag_embed(logvar.exp()))
        mix = torch.distributions.Categorical(torch.ones(mean.shape[:-1], ))
        dist_q = torch.distributions.MixtureSameFamily(mix, dist_qx)  # gmm

    H_p = dist_p.entropy().mean(dim=0)
    H_q = dist_q.entropy()
    mutual_info = - (H_q - H_p).sum()

    to_report, for_metrics = {}, {}
    return mutual_info, to_report, for_metrics


class SaveTensors(torchmetrics.Metric):
    """
    Generic class that is used to save tensors during prediction, so that they are available at the end of the loop
    Useful for visualizations
    """

    def __init__(self, num_tensors=1):
        super().__init__()

        self.add_state("tensors", default=[])
        self.num_tensors = None
        self.tensor_names = None

    def update(self, tensors: list):
        if self.num_tensors is None:
            self.num_tensors = len(tensors)
        else:
            assert len(tensors) == self.num_tensors
        self.tensors.append(tensors)

    def compute(self, clear_after=True):
        to_return_tensors = [[] for _ in range(self.num_tensors)]
        for i in range(len(self.tensors)):
            for j in range(self.num_tensors):
                to_return_tensors[j].append(self.tensors[i][j])
        for j in range(self.num_tensors):
            to_return_tensors[j] = torch.cat(to_return_tensors[j])

        if clear_after:
            self.clear()
        return to_return_tensors

    def clear(self):
        """
        Delete the tensors at the end. Otherwise there are synchronization problems because it is a list.
        Another option would be to change the dist_reduce_fx or let it be a tensor.
        """
        self.tensors.clear()


class SyncFunction(torch.autograd.Function):
    """
    Auxiliary class to concatenate tensors from different machines. This is necessary when the loss requires all the
    elements to be used (this is, there is interaction between batch samples), for example for contrastive losses
    """

    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]

    @classmethod
    def get_index(cls, tensor):
        device_v1 = int(os.environ.get("LOCAL_RANK", 0))
        device_v2 = tensor.device.index
        assert device_v1 == device_v2, f'Is the tensor in CUDA? We got devices {device_v1} and {device_v2}'
        indices = torch.ones(tensor.shape[0]).to(tensor.device) * device_v1
        indices_all = cls.apply(indices)
        assert not indices_all.requires_grad
        return indices_all == device_v1


class FuturePrediction(torchmetrics.Metric):
    """
    Future prediction distances. We report the mean as well as the mean-per-step. For each example, we use the best
    out of k (which is computed point-wise, not trajectory-wise)
    """
    distances: torch.Tensor
    total: torch.Tensor

    def __init__(self, k=10, distance_fn_name='euclidean_l2', param_dist=None, prediction_key='seg_dec_f_inp_p',
                 ground_truth_key='seg_inp_f'):
        super().__init__()

        max_steps = 100  # Upper bound on max_steps
        self.add_state("distances", default=torch.zeros(max_steps), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros(max_steps), dist_reduce_fx="sum")
        self.k = k
        self.distance_fn_name = distance_fn_name
        self.param_dist = param_dist
        self.prediction_key = prediction_key
        self.ground_truth_key = ground_truth_key

    def update(self, all_dicts=None, query=None, target=None, duration=None, **kwargs):
        if all_dicts is not None:
            assert self.prediction_key in all_dicts, \
                f'To compute FuturePrediction we need {self.prediction_key}. ' \
                f'Options like extrapolation may be necessary'
            prediction = all_dicts[self.prediction_key]['tensor']
            ground_truth = all_dicts[self.ground_truth_key]['tensor']
            duration = all_dicts[self.ground_truth_key]['seg_len']
        else:
            prediction = query
            ground_truth = target

        dist_fn = distances.get_dist_fn(self.distance_fn_name, self.param_dist)
        same_leading_dims = np.minimum(len(ground_truth.shape), len(prediction.shape)) - 1
        dists = dist_fn(prediction, ground_truth, same_leading_dims=same_leading_dims)
        # self.k is only defined here for the assertion. It has to be defined in the config with num_sample_points
        # assert dists.shape[-1] == self.k, f'Set num_sample_points to {self.k}'
        dists = dists.min(-1)[0]  # Compute best out of all samples

        if same_leading_dims == 2:
            # Still need to filter by time
            mask = torch.arange(dists.shape[1])[None, :].to(duration.device) < duration[:, None]
            dists = dists * mask
            self.distances[:dists.shape[1]] += dists.sum(0).cpu()
            self.total[:dists.shape[1]] += mask.sum(0).cpu()

        else:
            self.distances[0] += dists.sum().cpu()
            self.total += dists.shape[0]

    def compute(self):
        all_steps = self.distances.sum() / self.total.sum()
        per_step = self.distances[self.total != 0] / self.total[self.total != 0]
        # per_step = all_steps
        return {'all_steps': all_steps, 'per_step': per_step}


class TrajectoryAccuracy(torchmetrics.Metric):
    """
    Trajectory retrieval accuracy
    """
    total: torch.Tensor

    def __init__(self, key='tra_inp_p', query='tra_inp_f', latent_distribution='gaussian',
                 distance_type='kl-divergence', num_latent_params=2, return_mr=False, k: list = None,
                 swap_inputs=False, n=500):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__()

        self.k = k if k is not None else [1]  # Recall@k
        self.swap_inputs = swap_inputs
        self.n = n

        for k_ in self.k:
            self.add_state(f"correct_{k_}", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        self.return_mr = return_mr  # Return median rank
        if return_mr:
            self.add_state("values_mr", default=[], dist_reduce_fx="cat")

        self.distance_type = distance_type
        self.latent_distribution = latent_distribution
        self.num_latent_params = num_latent_params
        self.key = key
        self.query = query

    def update(self, all_dicts, **kwargs):

        dist_fn = distances.get_dist_fn(self.latent_distribution)

        if self.key not in all_dicts or self.query not in all_dicts:
            return
        query, target = all_dicts[self.key]['tensor'], all_dicts[self.query]['tensor']

        if len(query.shape) == 3:  # option_reencode == 2, and we are dealing with some reencoded trajectory
            query = query[:, 0]  # As if option_reencode was 1
        if len(target.shape) == 3:
            target = target[:, 0]

        # All samples are used as negatives
        # Combine values from all samples
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            query = SyncFunction.apply(query.contiguous())
            target = SyncFunction.apply(target.contiguous())

        # --------------------- Compute scores ------------------------ #
        param_1_query, param_2_query = distances.get_params(query, self.num_latent_params)
        param_1_target, param_2_target = distances.get_params(target, self.num_latent_params)

        param_1_query = param_1_query[:, None]
        param_2_query = param_2_query[:, None] if param_2_query is not None else None
        param_1_target = param_1_target[None, :]
        param_2_target = param_2_target[None, :] if param_2_target is not None else None

        dist_matrix = dist_fn(param_1_query, param_1_target, param_2_query, param_2_target,
                              distance_type=self.distance_type)
        dist_matrix = dist_matrix.permute(-2, -1)
        scores = -dist_matrix
        # ------------------------------------------------------------- #

        if scores.isnan().any():
            print('Some of the scores are nan() during TrajectoryAccuracy evaluation')
            return
        # self.n-1 is the number of samples to be used as negatives. Fixed so that value does not depend on batch size
        if target.shape[0] < self.n - 1:  # Maybe it's simply the last batch
            print('You may want to increase the total batch size, or reduce self.n')
            return  # Do not update.

        # Prepare potential negatives
        matrix_negatives = 1 - torch.eye(*scores.shape)
        indices_negatives = torch.multinomial(matrix_negatives, self.n - 1, replacement=False)

        # Add the positive
        indices_use = torch.cat([torch.arange(indices_negatives.shape[0])[:, None], indices_negatives], dim=1) \
            .to(target.device)
        scores = torch.gather(scores, -1, indices_use)
        gt = torch.zeros(indices_negatives.shape[0]).to(target.device)

        for k_ in self.k:
            if scores.shape[-1] < k_:
                k_ = 1
                if not hasattr(self, 'correct_1'):
                    self.add_state(f"correct_1", default=torch.tensor(0), dist_reduce_fx="sum")
            pred = torch.topk(scores, k_, dim=-1)
            setattr(self, f'correct_{k_}',
                    getattr(self, f'correct_{k_}') + (pred.indices == gt.unsqueeze(-1)).any(-1).sum())
        # -1 are the indices corresponding to zero padding
        self.total += torch.tensor(gt[gt != -1].numel())

        if self.return_mr:
            order = torch.argsort(-scores, -1)
            position = (gt.unsqueeze(-1) == order).float()[gt != -1].argmax(-1)
            position = position + 1
            self.values_mr += list(position)

    def compute(self):
        if self.total == 0:  # May be last batch
            to_return = {f'acc_{k_}': -1 for k_ in self.k}
            if self.return_mr:
                to_return['mr'] = -1
            return to_return
        # compute final result. This already combines multiple gpus (using dist_reduce_fx)
        results = {f'acc_{k_}': getattr(self, f'correct_{k_}').float() / self.total for k_ in self.k}
        if self.return_mr:
            values_mr = torch.stack(self.values_mr) if type(self.values_mr) == list else self.values_mr
            results['mr'] = values_mr.median() if len(self.values_mr) > 0 else -1
        return results


class TrajectoryAccuracyHard(torchmetrics.Metric):
    """
    Trajectory retrieval accuracy, but the negatives come from the same trajectory
    """
    total: torch.Tensor
    correct: torch.Tensor

    def __init__(self, key='tra_inp_p', query='tra_inp_f', negative='', latent_distribution='gaussian',
                 distance_type='kl-divergence', num_latent_params=2):

        super().__init__()

        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")

        self.distance_type = distance_type
        self.latent_distribution = latent_distribution
        self.num_latent_params = num_latent_params
        self.key = key
        self.query = query
        self.negative = negative

    def update(self, all_dicts, **kwargs):

        dist_fn = distances.get_dist_fn(self.latent_distribution)

        if self.key not in all_dicts or self.query not in all_dicts or self.negative not in all_dicts:
            return
        query = all_dicts[self.key]['tensor']
        target = all_dicts[self.query]['tensor']
        negative = all_dicts[self.negative]['tensor']

        if len(query.shape) == 3:  # option_reencode == 2, and we are dealing with some reencoded trajectory
            query = query[:, 0]  # As if option_reencode was 1
        if len(target.shape) == 3:
            target = target[:, 0]
        if len(negative.shape) == 3:
            negative = negative[:, 0]

        # --------------------- Compute scores ------------------------ #
        param_1_query, param_2_query = distances.get_params(query, self.num_latent_params)
        param_1_target, param_2_target = distances.get_params(target, self.num_latent_params)
        param_1_negative, param_2_negative = distances.get_params(negative, self.num_latent_params)

        distances_positive = dist_fn(param_1_query, param_1_target, param_2_query, param_2_target,
                                     distance_type=self.distance_type)
        distances_negative = dist_fn(param_1_query, param_1_negative, param_2_query, param_2_negative,
                                     distance_type=self.distance_type)

        self.total += distances_positive.shape[0]
        self.correct += (distances_positive < distances_negative).sum().item()

    def compute(self):
        if self.total == 0:  # May be last batch
            to_return = {'acc': -1}
            return to_return
        # compute final result. This already combines multiple gpus (using dist_reduce_fx)
        results = {'acc': getattr(self, 'correct').float() / self.total}
        return results


def categorical_softmax(x, dim=-1):
    # For the KL we have to make sure the prior does not have any value equal to zero. To do that, we artifiically
    # limit the values it can take. For consistency, we do the same everywhere
    x = torch.sigmoid(x)
    x = torch.softmax(x, dim=dim)
    return x
