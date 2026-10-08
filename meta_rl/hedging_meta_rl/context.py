"""Context for the paper option-hedging pipeline."""

from __future__ import annotations
from collections.abc import Sequence
import torch
from torch import nn
from torch.nn import functional as F
from rl_agents import _get_mlp


class ProbabilisticContextEncoder(nn.Module):
    """Infer Gaussian task posteriors by combining transition-level Gaussian factors with a standard normal prior."""

    def __init__(
        self,
        *,
        num_context_feature: int = 16,
        latent_dimension: int = 4,
        hidden_dims: Sequence[int] = (128, 128),
        activation_name: str = "relu",
        minimum_variance: float = 1e-06,
    ) -> None:
        """Initialize validated configuration and internal state."""
        super().__init__()
        if isinstance(num_context_feature, bool) or int(num_context_feature) <= 0:
            raise ValueError("num_context_feature must be a strictly positive integer")
        if isinstance(latent_dimension, bool) or int(latent_dimension) <= 0:
            raise ValueError("latent_dimension must be a strictly positive integer")
        if not torch.isfinite(torch.as_tensor(float(minimum_variance))):
            raise ValueError("minimum_variance must be a finite number")
        if float(minimum_variance) <= 0.0:
            raise ValueError("minimum_variance must be strictly positive")
        self.num_context_feature = int(num_context_feature)
        self.latent_dimension = int(latent_dimension)
        self.hidden_dims = tuple((int(value) for value in hidden_dims))
        self.activation_name = str(activation_name).lower()
        self.minimum_variance = float(minimum_variance)
        self.network = _get_mlp(
            self.num_context_feature,
            2 * self.latent_dimension,
            self.hidden_dims,
            self.activation_name,
        )

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute network outputs for the supplied tensor batch."""
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context.ndim != 3 or context.shape[-1] != self.num_context_feature:
            raise ValueError(
                f"context shape must be (L,C) or (B,L,C), and C={self.num_context_feature}"
            )
        num_batch, num_context, _ = context.shape
        if num_context == 0:
            mean = torch.zeros(
                (num_batch, self.latent_dimension), dtype=context.dtype, device=context.device
            )
            variance = torch.ones_like(mean)
            return (mean, variance)
        precision_sum, precision_weighted_mean_sum = self.get_sufficient_statistics(context)
        return self.get_posterior_from_sufficient_statistics(
            precision_sum, precision_weighted_mean_sum
        )

    def get_sufficient_statistics(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return sufficient statistics."""
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context.ndim != 3 or context.shape[-1] != self.num_context_feature:
            raise ValueError(
                f"context shape must be (L,C) or (B,L,C), and C={self.num_context_feature}"
            )
        num_batch, num_context, _ = context.shape
        if num_context == 0:
            zeros = torch.zeros(
                (num_batch, self.latent_dimension), dtype=context.dtype, device=context.device
            )
            return (zeros, zeros.clone())
        factors = self.network(context)
        factor_mean, raw_factor_variance = factors.chunk(2, dim=-1)
        factor_variance = F.softplus(raw_factor_variance) + self.minimum_variance
        factor_precision = factor_variance.reciprocal()
        return (factor_precision.sum(dim=1), (factor_precision * factor_mean).sum(dim=1))

    @staticmethod
    def get_posterior_from_sufficient_statistics(
        precision_sum: torch.Tensor, precision_weighted_mean_sum: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return posterior from sufficient statistics."""
        if precision_sum.ndim != 2 or precision_sum.shape != precision_weighted_mean_sum.shape:
            raise ValueError(
                "Both sufficient statistics must be two-dimensional tensors of equal shape"
            )
        if torch.any(precision_sum < 0.0):
            raise ValueError("precision_sum must be nonnegative")
        posterior_variance = (1.0 + precision_sum).reciprocal()
        posterior_mean = posterior_variance * precision_weighted_mean_sum
        return (posterior_mean, posterior_variance)

    @staticmethod
    def get_kl_divergence(mean: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
        """Return KL divergence."""
        if mean.shape != variance.shape or mean.ndim != 2:
            raise ValueError("mean and variance must be two-dimensional tensors of equal shape")
        if torch.any(variance <= 0):
            raise ValueError("posterior variance must be strictly positive")
        return 0.5 * (mean.square() + variance - torch.log(variance) - 1.0).sum(dim=-1)

    @staticmethod
    def sample(
        mean: torch.Tensor, variance: torch.Tensor, *, is_deterministic: bool
    ) -> torch.Tensor:
        """Sample."""
        if mean.shape != variance.shape:
            raise ValueError("mean and variance must have equal shapes")
        if is_deterministic:
            return mean
        return mean + variance.sqrt() * torch.randn_like(mean)

    def get_config(self) -> dict[str, object]:
        """Return the configuration required to reconstruct this object."""
        return {
            "num_context_feature": self.num_context_feature,
            "latent_dimension": self.latent_dimension,
            "hidden_dims": list(self.hidden_dims),
            "activation_name": self.activation_name,
            "minimum_variance": self.minimum_variance,
        }
