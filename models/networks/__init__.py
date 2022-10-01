import importlib
import sys

from torch import nn

from models.networks.baselines_networks import *
from models.networks.implicit import *
from models.networks.positional_encodings import *
from models.networks.st_gcn import ST_GCN_18
from models.networks.transformers import *


def get_network(name: str, **kwargs) -> nn.Module:
    if name is None:
        return nn.Identity()
    if name.startswith('torch_'):
        return getattr(importlib.import_module('torchvision.models'), name.replace('torch_', ''))
    else:
        return getattr(sys.modules[__name__], name)(**kwargs)
