from typing import Tuple

import torch
from box_embeddings.modules.intersection._intersection import _Intersection
from box_embeddings.parameterizations.box_tensor import BoxTensor


def _compute_logaddexp(
        t1: BoxTensor, t2: BoxTensor, enclosing_temperature: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    t1_data = torch.stack((-t1.z, t1.Z), -2)
    t2_data = torch.stack((-t2.z, t2.Z), -2)
    lse = torch.logaddexp(
        t1_data / enclosing_temperature, t2_data / enclosing_temperature
    )

    z = -enclosing_temperature * lse[..., 0, :]
    Z = enclosing_temperature * lse[..., 1, :]

    return z, Z


def gumbel_enclosing(
        left: BoxTensor,
        right: BoxTensor,
        enclosing_temperature: float = 1.0
) -> BoxTensor:
    """
    See gumbel_intersection in box_embeddings.modules.intersection.gumbel_intersection
    """

    t1 = left
    t2 = right

    if enclosing_temperature == 0:
        raise ValueError("enclosing_temperature must be non-zero.")

    z, Z = _compute_logaddexp(t1, t2, enclosing_temperature)

    return left.from_zZ(z, Z)


def hard_enclosing(left: BoxTensor, right: BoxTensor) -> BoxTensor:
    """Hard Enclosing operation as a function."""
    t1 = left
    t2 = right
    z = torch.min(t1.z, t2.z)
    Z = torch.max(t1.Z, t2.Z)

    return left.from_zZ(z, Z)


class Enclosing(_Intersection):  # Inheriting from _Intersection allows us to reuse its broadcasting
    """All for one Enclosing operation as Layer/Module"""

    def __init__(
            self,
            enclosing_temperature: float = 0.0
    ) -> None:

        super().__init__()  # type: ignore
        self.enclosing_temperature = enclosing_temperature

    def _forward(self, left: BoxTensor, right: BoxTensor) -> BoxTensor:
        """Gives the smallest enclosing box of two boxes."""
        if self.enclosing_temperature == 0:
            return hard_enclosing(left, right)
        else:
            return gumbel_enclosing(
                left,
                right,
                self.enclosing_temperature,
            )
