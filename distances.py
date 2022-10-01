"""
Set of functions that compute distances
"""

from itertools import chain

import torch
from box_embeddings.modules.intersection import Intersection
from box_embeddings.modules.volume import Volume
from box_embeddings.parameterizations.box_tensor import BoxTensor

from utils.utils import my_autocast
from utils.utils_bboxes import Enclosing


def my_sqrt(x):
    """
    The derivative of the square root at zero is not defined. Most of the time, it is not a problem but sometimes
    Pytorch returns nan, so here I just apply it to the elements that are not zero (the other elements stay the same)
    """
    x[x != 0] = x[x != 0].sqrt()
    return x


def get_dist_fn(distance_name, param_dist=None):
    """
    Both for point-to-point and distribution ones
    :param param_dist:
    :param distance_name:
    :return:
    """

    dict_distances = {
        'euclidean_l2': euclidean_l2,
        'euclidean_l1': euclidean_l1,
        'euclidean_l2_keypoints': euclidean_l2_keypoints,
        'euclidean_l1_keypoints': euclidean_l1_keypoints,
        'bbox': bbox_distance,
        'gaussian': gaussian,
        'laplacian': laplacian
    }
    return dict_distances[distance_name]


def get_prob_fn(distance_name):
    dist_fn = get_dist_fn(distance_name)

    def prob_fn_(param_1, param_2, point):
        return -dist_fn(param_1, point, param_2, distance_type='log-likelihood')

    return prob_fn_


# ---------------------------------- Point to point distances ------------------------------ #

def euclidean_l2(x, y, same_leading_dims=0, remove_diag=False, **kwargs):
    def distance_fn(x_, y_, **kwargs_):
        d = my_sqrt((x_ - y_).pow(2).sum(-1))
        return d

    dist = compute_distance(distance_fn, x, y, same_leading_dims, remove_diag)

    return dist


def euclidean_l1(x, y, same_leading_dims=0, remove_diag=False, **kwargs):
    def distance_fn(x_, y_, **kwargs_):
        d = (x_ - y_).abs().sum(-1)
        return d

    dist = compute_distance(distance_fn, x, y, same_leading_dims, remove_diag)

    return dist


def euclidean_l2_keypoints(x, y, same_leading_dims=0, remove_diag=False, **kwargs):
    """
    Same as euclidean_l2, but the keypoints that are zero are ignored. Also every keypoint in the sample is treated
    independently (not as a large 50D pont)
    """

    def distance_fn(x_, y_, **kwargs_):
        y_ = y_.view(*y_.shape[:-1], 25, 2)
        x_ = x_.view(*x_.shape[:-1], 25, 2)
        mask = y_.mean(-1) != 0
        d = my_sqrt((x_ - y_).pow(2).sum(-1))
        d = (d * mask).sum(-1) / mask.sum(-1)
        return d

    dist = compute_distance(distance_fn, x, y, same_leading_dims, remove_diag)

    return dist


def euclidean_l1_keypoints(x, y, same_leading_dims=0, remove_diag=False, **kwargs):
    """
    Same as euclidean_l1, but the keypoints that are zero are ignored. Also every keypoint in the sample is treated
    independently (not as a large 50D pont)
    """

    def distance_fn(x_, y_, **kwargs_):
        y_ = y_.view(*y_.shape[:-1], 25, 2)
        x_ = x_.view(*x_.shape[:-1], 25, 2)
        mask = y_.mean(-1) != 0
        d = (x_ - y_).abs().sum(-1)
        d = (d * mask).sum(-1) / mask.sum(-1)
        return d

    dist = compute_distance(distance_fn, x, y, same_leading_dims, remove_diag)

    return dist


def compute_distance(distance_fn, x, y, same_leading_dims=0, remove_diag=False, **kwargs):
    """
    Given two vectors x and y, it returns the pairwise distance matrix between all of them.
    :param distance_fn: distance metric
    :param x: [..., N, D]
    :param y: [..., N, D]
    :param same_leading_dims: number of dimensions that are equivalent for x and y (no cross-computing of distance)
    :param remove_diag: if x and y contain the same points, the diagonal of the distance will be comparing the same
    points (so distance = 0), and gradient becomes NaN. Therefore, we have two avoid computing those zeros. Things that
    do not work:
        - Any inplace operation (like dist[., .] = 0) will not work because of gradients
        - Just multiplying by a mask (dist = dist * mask) also doesn't work because gradients are still computed
        - Anything that involves computing the distance, even if then it is not used like dist.triu(1) + dist.tril(-1)
            (note that previous two points also fall into this, so they don't work for two different reasons)
    :return: distance matrix [..., N, N]
    """
    assert x.shape[:same_leading_dims] == y.shape[:same_leading_dims]
    len_x_shape = len(x.shape)
    for i in range(len(y.shape) - 1 - same_leading_dims):
        x = x.unsqueeze(-2)
    for i in range(len_x_shape - 1 - same_leading_dims):
        y = y.unsqueeze(same_leading_dims)

    if remove_diag and torch.is_grad_enabled():
        dist = torch.zeros((*x.shape[:-3], x.shape[-3], y.shape[-2])).to(x.device)
        triu_indices = torch.triu_indices(x.shape[-3], y.shape[-2], 1)  # Upper triangular (not including diagonal)
        indices_x = list(chain(*[(x.shape[-3] - 1 - i) * [i] for i in range(x.shape[-3])]))
        indices_y = list(chain(*[list(range(i + 1, y.shape[-2])) for i in range(y.shape[-2] - 1)]))
        x_ = x[..., indices_x, 0, :]
        y_ = y[..., 0, indices_y, :]
        dist_ = distance_fn(x_, y_, **kwargs)
        dist[..., triu_indices[0], triu_indices[1]] = dist_
        dist = dist + dist.transpose(-2, -1)
    else:
        dist = distance_fn(x, y, **kwargs)

    return dist


# -------------------------------- Distribution distances --------------------------- #
intersection_temperature = 0.01
volume_temperature = 0.1
enclosing_fn = Enclosing(enclosing_temperature=intersection_temperature)
intersection_fn = Intersection(intersection_temperature=intersection_temperature)
volume_fn = Volume(volume_temperature=volume_temperature, intersection_temperature=intersection_temperature)


@my_autocast(back_to_half=False)
def bbox_distance(box_a_z, box_b_z, box_a_Z, box_b_Z, distance_type='iou', log_space=True, **kwargs):
    """
    Computes the distance between two boxes. In order for these distances to be useful as probabilities for the bce
    loss, we normalize them in the [0, 1] range (when it is natural for them -- all of them except for 'euclidean'
    and 'sym_diff')

    :param log_space: return the -logarithm of the similarity (default: True), obtaining ranges [0, inf). It is the
        natural way these distances are computed, and the expected input to the BCE loss
        NOTE: when we use distances in the log space, we return -log(p) [in cases where the similarity can be understood
            as a probability, BUT when we return the exponential version (log_space==False), we return 1-p, which is NOT
            exp(-log(p)). We do this because the distance in scale [0, 1] makes more sense (when log_space==False), but
            in the logarithmic version, -log(p) works better than log(1-p).
    :param box_a_z: 
    :param box_b_z: 
    :param box_a_Z: 
    :param box_b_Z: 
    :param distance_type: there are different distance metrics we can use between bboxes:
        - 'iou': intersection over union.
        - 'euclidean': Average of euclidean distances between extremes of the two boxes
        - 'prediction': standard loss in the box embedding papers, where we predict p(A|B)
        - 'enclosing': similarity function S such that if we maximize S(A,R)+S(B,R) for some box R, it yields a box R
            which is the smallest enclosing box for A and B [not proven]. S(A, B) = P(A | B) + P(B | A)
        - 'sym_diff': Symmetric difference (union - intersection)
    :param kwargs:
    :return: 
    """
    assert distance_type in ['iou', 'euclidean', 'prediction', 'enclosing', 'sym_diff']
    assert not (distance_type == 'euclidean' and not log_space), \
        'log_space does not mean anything here, so the default of not doing anything is what we want (log_space=True)'

    if distance_type == 'euclidean':
        dist_z = euclidean_l2(box_a_z, box_b_z)
        dist_Z = euclidean_l2(box_a_Z, box_b_Z)
        dist = (dist_z + dist_Z) / 2
        return dist

    box_a = BoxTensor(torch.stack([box_a_z, box_a_Z], dim=-2))
    box_b = BoxTensor(torch.stack([box_b_z, box_b_Z], dim=-2))
    box_inter = intersection_fn(box_a, box_b)
    vol_a = volume_fn(box_a)
    vol_b = volume_fn(box_b)
    vol_inter = volume_fn(box_inter)

    if distance_type == 'iou':
        """
        Implement this with PyTorch logsumexp. Because there is a negative, we need to pass a scalar to logsumexp,
        which is currently being implemented. Should be done very soon. Track this:
        https://github.com/pytorch/pytorch/pull/71870
        """
        # vol_union = torch.logsumexp(torch.stack([vol_a,  vol_b,  vol_inter], w = [1, 1, -1], dim=0), dim=0)
        # Provisional stable implementation
        c1 = torch.max(torch.max(vol_a, vol_b), vol_inter)
        vol_union = ((vol_a - c1).exp() + (vol_b - c1).exp() - (vol_inter - c1).exp()).log() + c1

        iou = vol_inter - vol_union  # log_iou, really. [-inf, 0)
        dist = - iou

    elif distance_type == 'prediction':
        # P(B | A), where B is the ground truth, and A is the prediction
        p_ba = vol_inter - vol_a
        dist = - p_ba  # dist in [0, inf)

    elif distance_type == 'enclosing':
        # same as 'prediction', but symmetric
        # S(A, B) = (P(A | B) + P(B | A)) / 2
        p_ab = vol_inter - vol_b
        p_ba = vol_inter - vol_a
        log_similarity = torch.logaddexp(p_ab, p_ba) - torch.log(torch.tensor(2))
        dist = - log_similarity  # dist in [0, inf)

    elif distance_type == 'sym_diff':
        # Implement with PyTorch logsumexp. See message above
        c = torch.max(torch.max(vol_a, vol_b), vol_inter)  # auxiliary variable
        # union - intersection
        symmetric_difference = ((vol_a - c).exp() + (vol_b - c).exp() - 2 * (vol_inter - c).exp()).log() + c
        dist = symmetric_difference  # symmetric_difference has range [0, inf], so the log has range [-inf, inf]

    else:
        raise KeyError

    if not log_space:
        dist = 1 - torch.exp(-dist)

    return dist


@my_autocast(device='cpu')  # some operations not implemented for Half in cpu
def gaussian(mean_a, mean_b, logvar_a=None, logvar_b=None, rho_space=None, rho_time=None, rho_space_b=None,
             rho_time_b=None, distance_type='log-likelihood', share_dims=False):
    """
    Computes the distance between two gaussian distributions (a and b). The distance may not be a proper distance (e.g.
    it can be a divergence). 
    Different distance options (distance_type):
        - log-likelihood
        - kl-divergence
        - optimal-transport
    
    logvar means log(sigma^2), not log(sigma)
    """

    assert distance_type in ['log-likelihood', 'kl-divergence', 'optimal-transport']

    if logvar_a is None:  # L2 regression. Same for log-likelihood, KL-divergence, and optimal transport
        dist = (mean_b - mean_a).pow(2).sum(-1)
        return dist

    if distance_type == 'optimal-transport':
        # Optimal Transport is implemented as Wasserstein L2.
        assert rho_space is None and rho_time is None, 'OT only implemented for uncorrelated multivariate Gaussians'
        # Mathematically this should always be >=0, but there may be some numerical precision issues, so we make sure,
        # in order not to have negative distances
        std_term = torch.relu(logvar_a.exp() + logvar_b.exp() - 2 * ((logvar_a + logvar_b) / 2).exp())
        distance_w = (mean_b - mean_a).pow(2) + std_term
        dist = distance_w.sum(-1)
        return dist

    # This is now only for log-likelihood or kl-divergence
    if rho_space is None and rho_time is None:
        if len(logvar_a.shape) == 2 and share_dims:
            logvar_a = logvar_a[..., None].repeat(1, 1, 2)  # The two dimensions have the same variance
        if distance_type == 'log-likelihood':
            # We add the log(2pi) to have the actual result value, but it is not necessary for optimization
            log_prob = - 1 / 2 * (torch.tensor(2 * torch.pi).log() * mean_a.shape[-1] +
                                  (logvar_a + (mean_b - mean_a).pow(2) / logvar_a.exp()).sum(-1))
            dist = - log_prob

        else:  # kl-divergence KL(B||A)
            dist = 1 / 2 * (logvar_a - logvar_b) - 1 / 2 + (logvar_b.exp() + (mean_a - mean_b).pow(2)) / (
                    2 * logvar_a.exp())
            dist = dist.clamp(min=0)  # in case there are numerical errors (mostly for half precision)
            dist = dist.sum(-1)  # product (in log space is a sum) of all dimensions
        return dist

    if rho_time is None and rho_space is not None:
        rho_space = torch.tanh(rho_space)  # [B, T] in range [-1, 1]
        log_det_var = 1 / 2 * (torch.log(1 - rho_space.pow(2)) + logvar_a.sum(-1))
        independent_term = ((mean_b - mean_a).pow(2) / logvar_a.exp()).sum(-1)
        correlation_term = 2 * rho_space * ((mean_b - mean_a) / (logvar_a / 2).exp()).prod(-1)

        if distance_type == 'log-likelihood':
            first_term = torch.tensor(2 * torch.pi).log() + log_det_var

        else:  # kl-divergence KL(B||A)
            log_det_var_gt = 1 / 2 * (torch.log(1 - rho_space_b.pow(2)) + logvar_b.sum(-1))
            first_term = log_det_var - log_det_var_gt - 1
            independent_term += (log_det_var_gt.exp() / log_det_var.exp()).sum(-1)
            correlation_term += 2 * rho_space_b * rho_space * ((logvar_b.sum(-1) - logvar_a.sum(-1)) / 2).exp()

        dist = first_term + 1 / (2 * (1 - rho_space.pow(2))) * (independent_term - correlation_term)

        return dist

    # rho_time is not None here
    assert distance_type == 'log-likelihood', "We don't compute KL divergence on distributions correlated in time, for now"
    # Full Gaussian
    std_pred = (logvar_a / 2).exp()  # [B, T, 2]
    rho_time = torch.tanh(rho_time)  # [B, T, T] in range [-1, 1]

    correlation_matrix = rho_time
    correlation_matrix = torch.tril(correlation_matrix, diagonal=-1)

    if rho_space:
        num_dims = mean_b.shape[-1] * mean_b.shape[-2]  # mean_b has shape [B, T, 2]
        correlation_matrix[:, range(mean_b.shape[-2]), range(mean_b.shape[-2])] = rho_space
        # Repeat for spatial dimensions
        correlation_matrix = correlation_matrix.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
        correlation_matrix[:, range(num_dims), range(num_dims)] = 1
        std_pred = std_pred.reshape(std_pred.shape[0], -1)  # join time and space
        mean_a = mean_a.reshape(mean_a.shape[0], -1)
        mean_b = mean_b.reshape(mean_b.shape[0], -1)
    else:  # only temporal correlation. Spatial dimensions are treated separately
        num_dims = mean_b.shape[-2]  # just time
        correlation_matrix = correlation_matrix.unsqueeze(-3).repeat(1, 2, 1, 1)
        std_pred = std_pred.transpose(-2, -1)  # [B, 2, T]
        mean_b = mean_b.transpose(-2, -1)
        mean_a = mean_a.transpose(-2, -1)

    i, j = torch.triu_indices(num_dims, num_dims)
    correlation_matrix[..., i, j] = correlation_matrix[..., j, i]
    correlation_matrix = correlation_matrix * std_pred.unsqueeze(-2) * std_pred.unsqueeze(-1)

    # correlation_matrix should have shape [B, 2T, 2T] if rho_space exists, [B, 2, T, T] otherwise

    correlation_matrix_inv = torch.linalg.inv(correlation_matrix)
    log_det_var = torch.linalg.det(correlation_matrix)
    time_dim = mean_b.shape[-2]
    dist = time_dim * torch.tensor(2 * torch.pi).log() - 1 / 2 * log_det_var - \
           1 / 2 * torch.matmul(torch.matmul((mean_b - mean_a).unsqueeze(-2), correlation_matrix_inv),
                                (mean_b - mean_a).unsqueeze(-1)).squeeze(-1).squeeze(-1)

    return dist


def laplacian(mean_a, mean_b, logvar_a=None, logvar_b=None, rho_space=None, rho_time=None, rho_space_b=None,
              rho_time_b=None, distance_type='log-likelihood'):
    """
    In the Laplacian, the parameters are the mean mu (location parameter), and the diversity (scale parameter) b
    We associate b to std in the Gaussian, and thus logvar is log(b) [the parameter b is unrelated to the variable b]

    Notes about Laplacian:
        - Unlike the multivariate normal distribution, even if the covariance matrix has zero covariance and correlation
        the variables are not independent.
        - Multivariate Laplacian distributions can be defined in at least three different manners.
        Therefore, we treat all dimensions as independent
    """
    if logvar_b is None:  # L1 regression
        dist = (mean_b - mean_a).abs().sum(-1)
        return dist

    if rho_space is None and rho_time is None:
        if len(logvar_a.shape) == 2:
            logvar_a = logvar_a[..., None].repeat(1, 1, 2)  # The two dimensions have the same variance
        if distance_type == 'log-likelihood':
            dist = (torch.log(torch.tensor(2)) + logvar_a + (mean_b - mean_a).abs() / logvar_a.exp()).sum(-1)
        elif distance_type == 'kl-divergence':
            dist = ((-(mean_b - mean_a).abs() / logvar_b.exp() + logvar_b).exp() +
                    (mean_b - mean_a).abs()) / logvar_a.exp() + logvar_a - logvar_b - 1
        else:
            raise KeyError(f'{distance_type} for Laplacians is not implemented')
        return dist

    else:
        raise NotImplementedError('Any kind of multivariate Laplacian operation is not implemented')


@my_autocast
def compute_intersection(a, b, point_trajectory, num_latent_params, latent_distribution):
    """
    Intersection of distributions is implemented as a product of the pdfs
    http://www.lucamartino.altervista.org/2003-003.pdf
    """
    param_1_a, param_2_a = get_params(a, num_latent_params)
    param_1_b, param_2_b = get_params(b, num_latent_params)

    if point_trajectory:
        assert param_2_a is None and param_2_b is None
        intersection = (param_1_a + param_1_b) / 2

    else:
        if latent_distribution == 'gaussian':
            assert num_latent_params == 2, 'Currently only implemented for (uncorrelated) Gaussians'
            logvar_a, logvar_b = param_2_a, param_2_b
            mean_a, mean_b = param_1_a, param_1_b
            var_intersection = torch.log((logvar_a + logvar_b).exp() / (logvar_a.exp() + logvar_b.exp()))
            mean_intersection = (logvar_a.exp() * mean_b + logvar_b.exp() * mean_a) / (logvar_a.exp() + logvar_b.exp())
            intersection = from_params(num_latent_params, mean_intersection, var_intersection)

        elif latent_distribution == 'bbox':
            box_a = BoxTensor(torch.stack([param_1_a, param_2_a], dim=-2))
            box_b = BoxTensor(torch.stack([param_1_b, param_2_b], dim=-2))
            box_inter = intersection_fn(box_a, box_b)
            intersection = from_params(num_latent_params, box_inter.z, box_inter.Z)

        else:
            raise KeyError

    return intersection


def compute_average(x, latent_distribution, num_latent_params, dim=-1):
    """
    Compute average of two distributions along the dimension dim.
    Note that this does NOT compute the distribution resulting from averaging two random variables samples from the
    input distributions. Rather, it returns the parameters for a distribution of the same family of distributions as the
    inputs. This is not necessarily well defined, so different options are possible.
    """
    param_1, param_2 = get_params(x, num_latent_params)
    if latent_distribution == 'gaussian':
        mean = param_1
        logvar = param_2
        mean_avg = mean.mean(dim=dim)
        # equal to logvar.exp().mean(dim=dim).log(), but computationally stable
        logvar_avg = torch.logsumexp(logvar, dim=dim) - torch.log(torch.tensor(logvar.shape[dim]))
        x_avg = from_params(num_latent_params, mean_avg, logvar_avg)

    elif latent_distribution == 'bbox':
        # Just regular average of extreme points
        z, Z = param_1, param_2
        x_avg = from_params(num_latent_params, z.mean(dim=dim), Z.mean(dim=dim))

    else:
        raise NotImplementedError

    return x_avg


def get_params(x, num_latent_params):
    """
    Keep separate to uniformize the way we structure the latent prediction into parameters (mean, variance for now)
    """
    if num_latent_params == 1:
        x_mean = x
        x_var = None
    elif num_latent_params == 2:
        # Could also implement as x.view(-1, 2), or x.view(2, -1). But this is fine
        x_mean = x[..., :x.shape[-1] // 2]
        x_var = x[..., x.shape[-1] // 2:]
    else:
        raise NotImplementedError
    return x_mean, x_var


def from_params(num_latent_params, param_1, param_2):
    """
    Counterpart of get_params
    """
    if num_latent_params == 1:
        tensor = param_1
    elif num_latent_params == 2:
        tensor = torch.cat([param_1, param_2], dim=-1)
    else:
        raise NotImplementedError
    return tensor
