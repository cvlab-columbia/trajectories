import functools
import hashlib
import os

import numpy as np
import omegaconf
import torch
import wandb
from pytorch_lightning.utilities.cloud_io import load as pl_load


def get_checkpoint_path(logger, resume):
    """
    Get the path of the checkpoint when loading some pretrained model.
    """
    if resume.id is None:  # Overwritten when id is not None
        checkpoint_path = resume.path
    else:
        # Note that resume.id will only be the same as logger.version if resume.load_all
        checkpoint_path = os.path.join(logger.save_dir, resume.dir_checkpoint, resume.id, 'checkpoints')
        options = os.listdir(checkpoint_path)
        if resume.epoch == 'best':
            all_possibilities = {name: float(name.split('val_loss=')[-1].replace('.ckpt', ''))
                                 for name in options if 'val_loss=' in name}
            name_ckpt = sorted(all_possibilities.items(), key=lambda item: item[1])[0][0]  # Lowest val loss
        elif resume.epoch is None or resume.epoch == 'last':
            name_ckpt = 'last.ckpt'
        else:  # a specific number
            name_ckpt = [option for option in options if f'epoch={resume.epoch}'][-1]
            if len(name_ckpt) == 0:
                raise Exception(f'A checkpoint on epoch {resume.epoch} does not exist')

        checkpoint_path = os.path.join(checkpoint_path, name_ckpt)
    return checkpoint_path


def load_model(model_class, checkpoint_path, model_cfg):
    """
    If we want to resume all the training state, this is done with `resume_from_checkpoint` in the Trainer
    The params used to create the model are the ones in the current config, not the saved ones
    While load_state and load_all have to be exactly the same model, in load_model we allow more flexibility, like not
    strict loading, and loading only submodules.
    """
    if type(checkpoint_path) == str:
        # Use default method to load checkpoint. Note that the config is the current one, not the saved one. Therefore,
        # the only difference wrt just loading state dict are callbacks that may be defined in model_class
        model = model_class.load_from_checkpoint(checkpoint_path, strict=False, **model_cfg)
    elif type(checkpoint_path) in [dict, omegaconf.dictconfig.DictConfig]:
        # Load submodules indicated by dict
        # Create model
        model = model_class(**model_cfg)
        # Load checkpoints for submodules
        for name_submodule, path_submodule in checkpoint_path.items():
            checkpoint_submodule = pl_load(path_submodule, map_location=lambda storage, loc: storage)
            # This is to load models from other frameworks, that use other keys for the model state dict
            model_state_keys = [key for key in ['state_dict', 'model_state', 'model'] if key in checkpoint_submodule]
            if len(model_state_keys) > 0:  # Otherwise, assume all the checkpoint is the model_state
                checkpoint_submodule = checkpoint_submodule[model_state_keys[0]]
            not_matching = model.get_submodule(name_submodule).load_state_dict(checkpoint_submodule, strict=False)
            print(not_matching)  # Just informative
    else:
        raise Exception('The checkpoint path has to be either string or dict.')

    return model


def check_same_config(config_new, logger):
    """
    Even when resuming an experiment with the same id, the a new wandb directory is created (a new run_...), that stores
    the information that is specific for that run (e.g. the hydra_config.yaml file can be different), even though in the
    web application all of it is together. Important: in the web application only the *last* of each file is kept. The
    logs are appended to the previous ones.
    """
    # First option: download hydra_config.yaml from the version stored online
    # It has the problem that it stores more info than the config, but we never compare it so it's ok
    api = wandb.Api()
    runs = api.runs(f"{config_new.wandb.entity}/{config_new.wandb.project}")
    run = [run for run in runs if run.id == logger.version][0]
    config_old = run.config

    # Second option (in case there is any problem with the first option): load config file from filesystem
    # Find the latest directory with the same id
    """
    list_runs_same_id = glob.glob(os.path.join(logger.save_dir, 'wandb', f'run-*-{logger.version}'))
    list_runs_same_id.sort()
    config_old = None
    for path_last_run in list_runs_same_id[::-1]:
        path_old_config = os.path.join(path_last_run, 'files', 'hydra_config.yaml')
        if os.path.isfile(path_old_config):  # Maybe there was an error before the hydra_config was saved.
            config_old = omegaconf.OmegaConf.load(path_old_config)
            break
    """

    to_compare = ['dataset', 'model']  # Other parameters can change, like how to checkpoint
    # Copy (to avoid comparing) the keys that we do not mind if they change, from to_compare. Only those that do not
    # change anything regarding reproducibility (does num_workers change randomness in the training loop?)
    config_old['dataset']['dataloader_params']['num_workers'] = \
        config_new['dataset']['dataloader_params']['num_workers']
    config_old['dataset']['split_use'] = config_new['dataset']['split_use']
    config_old['model']['predict_mode'] = config_new['model']['predict_mode']

    for key in to_compare:
        if config_new[key] != config_old[key]:
            m = f'(At least) the {key} config is different. '
            # Check if different keys
            diff_keys = set(config_new[key].keys()) - set(config_new[key].keys())
            if len(diff_keys) > 0:
                m += f'These keys are only in one of the configs: {config_new[key]}. '
            else:
                for k, v_new in config_new[key].items():
                    v_old = config_old[key][k]
                    if v_new != v_old:
                        m += f'The {key}.{k} values are different (new={v_new}, old={v_old}). '
            raise Exception(m)

    return


def random_derangement(n):
    """
    A derangement is a permutation of a sequence where not element is in the same position as before.
    This method returns a derangement of the sequence [1, ..., n]
    """
    while True:
        v = np.arange(n)
        for j in np.arange(n - 1, -1, -1):
            p = np.random.randint(0, j + 1)
            if v[p] == j:
                break
            else:
                v[j], v[p] = v[p], v[j]
        else:
            if v[0] != 0:
                return v


def compress_indices_(len_max, x_len):
    lead_dims = len(x_len.shape)
    indices = torch.arange(len_max)
    for i in range(lead_dims):
        indices = indices.unsqueeze(0)
    indices = indices.expand(*x_len.shape, len_max)
    indices = indices.to(x_len.device) < x_len[..., None]
    return indices


def compress_tensor(x, x_len):
    """
    Given a tensor x [B, N, ...], and some lengths [B] <= N, it returns a tensor [B*sum(x_len), ...].

    The use case is when each sample represents a sequence of length N, but only a few of those elements (x_len[i])
    are useful, the rest are padding. If we want to process the elements separately (not as a sequence), we do not need
    to process the zero paddings. The result of the processing can be reshaped again using "decompress_tensor"

    If len(x_len.shape) > 1, we assume x has several "batch" dimensions, and they match with x_len
    """
    lead_dims = len(x_len.shape)
    assert x_len.max() <= x.shape[lead_dims]
    indices = compress_indices_(x.shape[lead_dims], x_len)
    x = x[indices]
    return x


def decompress_tensor(x, x_len, len_max):
    """
    See explanation in compress_tensor

    x: [B*sum(x_len), ...]
    x_len: B
    len_max: maximum length of the sequence (N)

    return: [B, N, ...]
    """
    if len_max is None:
        len_max = x_len.max()
    assert x.shape[0] == x_len.sum()

    out = torch.zeros(x_len.shape[0], len_max, *x.shape[1:]).to(x.device).type(x.type())
    indices = compress_indices_(len_max, x_len)
    out[indices] = x

    return out


def my_autocast(func=None, *, back_to_half=True, device='cuda'):
    """
    Used for functions that require torch.tensor parameters to not be Half. The half parameters are moved to float, and
    then the output is  moved back to half (if back_to_half)
    Does not consider all cases. For example, if an input is a BoxTensor or a dictionary with attributes that are
    torch tensors, those will not be converted.
    Also, if the output is a dictionary, those outputs will not be converted back

    To be used either like (uses default back_to_half=True):
    @my_autocast
    def function(...)

    or

    @my_autocast(back_to_half=True/False)
    def function(...)

    The behavior of torch.autocast is similar, but it also enforces the operations within to always stay the specified
    type. So in general it is better, but this function I get to modify for specific purposes
    """
    if func is None:
        return functools.partial(my_autocast, back_to_half=back_to_half, device=device)

    @functools.wraps(func)
    def inner(*args, **kwargs):
        is_half = False
        name_type = 'torch.cuda.HalfTensor' if device == 'cuda' else 'torch.HalfTensor'
        new_args = []
        for a in args:
            if type(a) == torch.Tensor and a.type() == name_type:
                new_args.append(a.float())
                is_half = True  # If one input is torch.half, all torch.tensor outputs will be torch.half
            else:
                new_args.append(a)
        new_kwargs = {}
        for k, v in kwargs.items():
            if type(v) == torch.Tensor and v.type() == name_type:
                new_kwargs[k] = v.float()
                is_half = True
            else:
                new_kwargs[k] = v
        result = func(*new_args, **new_kwargs)
        if is_half and back_to_half:
            if type(result) == list:
                new_result = []
                for r in result:
                    if type(result) == torch.Tensor and result.type() == name_type:
                        result.append(r.half())
                    else:
                        result.append(r)
                result = new_result
            else:
                if type(result) == torch.Tensor and result.type() == name_type:
                    result = result.half()
        return result

    return inner


def logsubexp(a, b):
    """
    Logarithm of the subtraction of exponentiations of the inputs.
    Like torch.logaddexp, but for subtraction
    """
    c = torch.max(a, b)
    torch.log(torch.exp(a - c) - torch.exp(b - c)) + c


Hash = hashlib.sha512
MAX_HASH_PLUS_ONE = 2 ** (Hash().digest_size * 8)


def str_to_probability(in_str):
    """Return a reproducible uniformly random float in the interval [0, 1) for the given string."""
    if type(in_str) != str:
        in_str = str(in_str)
    seed = in_str.encode()
    hash_digest = Hash(seed).digest()
    hash_int = int.from_bytes(hash_digest, 'big')  # Uses explicit byteorder for system-agnostic reproducibility
    return hash_int / MAX_HASH_PLUS_ONE  # Float division
