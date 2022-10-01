"""
This file contains definitions of auxiliary functions for Hydra configurations. They are useful to define parameters
as a function of other configuration parameters
"""

from typing import Any, Union, Dict

from torch import nn

from .utils import *
from .wandb_logger import *


def if_equal_else(condition, target_condition, val_if, val_else):
    """
    Checks a condition. If the condition is met, returns val_if, otherwise returns val_else.
    The condition can be specified as a boolean (if target_condition is None), or as a value that has to be equal to
    target_condition (if target_condition exists)
    """
    if target_condition is None:
        if condition is None or \
                (type(condition) == str and condition == '') or \
                (type(condition) == float and condition == 0.) or \
                (type(condition) == int and condition == 0):
            return val_if
        else:
            return val_else
    if condition == target_condition:
        return val_if
    else:
        return val_else


def str_to_float(x: str):
    """
    Evaluates the expression x and returns a number. For example, x could be the string "3*2/5"
    """
    return eval(x)


def assert_value(given_value, target_value):
    """
    Used when we use the value from some other parameter (e.g. in the model we use parameter coming from dataset), but
    we have some specific value we need for the model).
    """
    assert given_value == target_value, f'Given value of {given_value} should be {target_value}'
    return given_value


ModuleFromConfig = Union[nn.Module, Dict[str, Any]]

omegaconf.OmegaConf.register_new_resolver("any", lambda *numbers: any(numbers))
omegaconf.OmegaConf.register_new_resolver("if", lambda condition, val: val if condition else None)
omegaconf.OmegaConf.register_new_resolver("product", lambda x, y: x * y)
omegaconf.OmegaConf.register_new_resolver("subtract", lambda x, y: x - y)
omegaconf.OmegaConf.register_new_resolver("if_equal_else", if_equal_else)
omegaconf.OmegaConf.register_new_resolver("assert_value", assert_value)
omegaconf.OmegaConf.register_new_resolver("str_to_float", str_to_float)
