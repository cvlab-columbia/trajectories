#!/usr/bin/env python3
# Adapted from https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/optimizer.py
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Optimizer."""

import math

import torch
from pytorch_lightning.callbacks.base import Callback


def construct_optimizer(model, cfg):
    """
    Construct a stochastic gradient descent or ADAM optimizer with momentum.
    Details can be found in:
    Herbert Robbins, and Sutton Monro. "A stochastic approximation method."
    and
    Diederik P.Kingma, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."

    Args:
        model (model): model to perform stochastic gradient descent
        optimization or ADAM optimization.
        cfg (config): configs of hyper-parameters of SGD or ADAM, includes base
        learning rate,  momentum, weight_decay, dampening, and etc.
    """

    model_list = [model] if type(model) != list else model

    bn_parameters = []
    non_bn_parameters = []
    zero_parameters = []
    no_requires_grad = []
    skip = {}
    for model in model_list:
        if hasattr(model, "no_weight_decay"):
            skip = model.no_weight_decay()

    for model in model_list:
        for name, m in model.named_modules():
            is_bn = isinstance(m, torch.nn.modules.batchnorm._NormBase)
            for p in m.parameters(recurse=False):
                if not p.requires_grad:
                    no_requires_grad.append(p)
                    continue
                if is_bn:
                    bn_parameters.append(p)
                elif name in skip or (
                        (len(p.shape) == 1 or name.endswith(".bias"))
                        and cfg.zero_wd_1d_param
                ):
                    zero_parameters.append(p)
                else:
                    non_bn_parameters.append(p)

    optim_params = [
        {"params": bn_parameters, "weight_decay": cfg.weight_decay},
        {"params": non_bn_parameters, "weight_decay": cfg.weight_decay},
        {"params": zero_parameters, "weight_decay": 0.0},
    ]
    optim_params = [x for x in optim_params if len(x["params"])]

    # Check all parameters will be passed into optimizer.
    len_model_params = len([p for model in model_list for p in list(model.parameters())])
    assert len_model_params == len(non_bn_parameters) + len(
        bn_parameters
    ) + len(
        zero_parameters
    ) + len(
        no_requires_grad
    ), "parameter size does not match: {} + {} + {} + {} != {}".format(
        len(non_bn_parameters),
        len(bn_parameters),
        len(zero_parameters),
        len(no_requires_grad),
        len_model_params,
    )
    print(
        "bn {}, non bn {}, zero {}".format(
            len(bn_parameters), len(non_bn_parameters), len(zero_parameters)
        )
    )

    if cfg.optimizing_method == "sgd":
        return torch.optim.SGD(
            optim_params,
            lr=cfg.base_lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            dampening=cfg.dampening,
            nesterov=cfg.nesterov,
        )
    elif cfg.optimizing_method == "adam":
        return torch.optim.Adam(
            optim_params,
            lr=cfg.base_lr,
            betas=(0.9, 0.999),
            weight_decay=cfg.weight_decay,
        )
    elif cfg.optimizing_method == "adamw":
        return torch.optim.AdamW(
            optim_params,
            lr=cfg.base_lr,
            eps=1e-08,
            weight_decay=cfg.weight_decay,
        )
    else:
        raise NotImplementedError(
            "Does not support {} optimizer".format(cfg.optimizing_method)
        )


def get_epoch_lr(cur_epoch, cfg):
    """
    Retrieves the lr for the given epoch (as specified by the lr policy).
    Args:
        cfg (config): configs of hyper-parameters of ADAM, includes base
        learning rate, betas, and weight decay.
        cur_epoch (float): the number of epoch of the current training stage.
    """
    return get_lr_at_epoch(cfg, cur_epoch)


def set_lr(optimizer, new_lr):
    """
    Sets the optimizer lr to the specified value.
    Args:
        optimizer (optim): the optimizer using to optimize the current network.
        new_lr (float): the new learning rate to set.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def get_lr_at_epoch(cfg, cur_epoch):
    """
    Retrieve the learning rate of the current epoch with the option to perform
    warm up in the beginning of the training stage.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    """
    lr = get_lr_func(cfg.lr_policy)(cfg, cur_epoch)
    # Perform warm up.
    if cur_epoch < cfg.warmup_epochs:
        lr_start = cfg.warmup_start_lr
        lr_end = get_lr_func(cfg.lr_policy)(
            cfg, cfg.warmup_epochs
        )
        alpha = (lr_end - lr_start) / cfg.warmup_epochs
        lr = cur_epoch * alpha + lr_start
    return lr


def lr_func_cosine(cfg, cur_epoch):
    """
    Retrieve the learning rate to specified values at specified epoch with the
    cosine learning rate schedule. Details can be found in:
    Ilya Loshchilov, and  Frank Hutter
    SGDR: Stochastic Gradient Descent With Warm Restarts.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    """
    offset = cfg.warmup_epochs if cfg.cosine_after_warmup else 0.0
    assert cfg.cosine_end_lr < cfg.base_lr
    return (
            cfg.cosine_end_lr
            + (cfg.base_lr - cfg.cosine_end_lr)
            * (
                    math.cos(
                        math.pi * (cur_epoch - offset) / (cfg.max_epoch - offset)
                    )
                    + 1.0
            )
            * 0.5
    )


def lr_func_steps_with_relative_lrs(cfg, cur_epoch):
    """
    Retrieve the learning rate to specified values at specified epoch with the
    steps with relative learning rate schedule.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    """
    ind = get_step_index(cfg, cur_epoch)
    return cfg.lrs[ind] * cfg.base_lr


def get_step_index(cfg, cur_epoch):
    """
    Retrieves the lr step index for the given epoch.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    """
    steps = cfg.steps + [cfg.max_epoch]
    for ind, step in enumerate(steps):  # NoQA
        if cur_epoch < step:
            break
    return ind


def get_lr_func(lr_policy):
    """
    Given the configs, retrieve the specified lr policy function.
    Args:
        lr_policy (string): the learning rate policy to use for the job.
    """
    policy = "lr_func_" + lr_policy
    if policy not in globals():
        raise NotImplementedError("Unknown LR policy: {}".format(lr_policy))
    else:
        return globals()[policy]


class LearningRateMonitorSlowFast(Callback):
    # def on_train_epoch_start(self, trainer, *args, **kwargs):
    def on_batch_start(self, trainer, model, **kwargs):
        # Call every step
        epoch = trainer.global_step  # Not actually an epoch, just a training step
        # Get actual model, not wrapper around it
        model = trainer.model
        while hasattr(model, 'module'):
            model = model.module
        current_lr = get_epoch_lr(epoch, model.optim_params)
        trainer.logger.log_metrics({'lr': current_lr}, step=epoch)
