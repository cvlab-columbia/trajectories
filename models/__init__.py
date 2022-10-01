import sys
from typing import Any, Dict, Tuple

from .rnn_model import VRNNModel
from .trajectory_model import TrajectoryModel


def get_model(name: str, **kwargs: Dict[str, Any]) -> Tuple[Any, Dict[str, Dict[str, Any]]]:
    return getattr(sys.modules[__name__], name), kwargs
