import math
from typing import List, Optional, Literal

import mlx.core as mx
import mlx.nn as nn


class PiecewiseLinearEmbeddings(nn.Module):
    """Piecewise-linear embeddings for MLX.

    Args:
        bins: List of bin edges for each feature.
        d_embedding: The embedding dimension.
        activation: Whether to apply ReLU activation.
        version: The version of the implementation ('A', 'B' or None).
    """

    def __init__(
        self,
        bins: List[mx.array],
        d_embedding: int,
        *,
        activation: bool = False,
        version: Literal[None, "A", "B"] = "B",
    ):
        if d_embedding <= 0:
            raise ValueError(f"d_embedding must be positive, got {d_embedding}")

        super().__init__()
        n_features = len(bins)

        # For backward compatibility
        if version is None:
            print("Warning: version not specified, defaulting to 'A'")
            version = "A"

        is_version_B = version == "B"

        # Version B has an additional linear embedding
        if is_version_B:
            self.linear0 = LinearEmbeddings(n_features, d_embedding)
        else:
            self.linear0 = None

        # The piecewise linear encoding implementation
        self.impl = _PiecewiseLinearEncodingImpl(bins)

        # The main linear transformation
        self.linear = _NLinear(
            len(bins),
            self.impl.get_max_n_bins(),
            d_embedding,
            bias=not is_version_B,
        )

        # Initialize weights for version B
        if is_version_B:
            self.linear.weight = mx.zeros_like(self.linear.weight)

        self.activation = nn.ReLU() if activation else None

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim != 2:
            raise ValueError("Input must have shape (batch_size, n_features)")

        x_linear = self.linear0(x) if self.linear0 is not None else None

        x_ple = self.impl(x)
        x_ple = self.linear(x_ple)

        if self.activation is not None:
            x_ple = self.activation(x_ple)

        return x_ple if x_linear is None else x_linear + x_ple


class _PiecewiseLinearEncodingImpl(nn.Module):
    """Implementation of piecewise linear encoding for MLX."""

    def __init__(self, bins: List[mx.array]):
        super().__init__()
        n_features = len(bins)
        n_bins = [len(x) - 1 for x in bins]
        max_n_bins = max(n_bins)

        # Initialize weights and biases
        self.weight = mx.zeros((n_features, max_n_bins))
        self.bias = mx.zeros((n_features, max_n_bins))

        # Handle single bin case
        single_bin_mask = mx.array([n == 1 for n in n_bins])
        self.single_bin_mask = (
            single_bin_mask if mx.any(single_bin_mask).item() else None
        )

        # Create mask for variable number of bins
        if not all(n == n_bins[0] for n in n_bins):
            mask = []
            for x in bins:
                n = len(x) - 1
                mask_row = mx.concatenate(
                    [
                        mx.ones((n - 1,), dtype=mx.bool_),
                        mx.zeros((max_n_bins - n,), dtype=mx.bool_),
                        mx.ones((1,), dtype=mx.bool_),
                    ]
                )
                mask.append(mask_row)
            self.mask = mx.stack(mask)
        else:
            self.mask = None

        # Set weights and biases based on bin edges
        for i, bin_edges in enumerate(bins):
            bin_width = bin_edges[1:] - bin_edges[:-1]
            w = 1.0 / bin_width
            b = -bin_edges[:-1] / bin_width

            # Last encoding component
            self.weight[i, -1] = w[-1]
            self.bias[i, -1] = b[-1]

            # Leading encoding components
            if n_bins[i] > 1:
                self.weight[i, : n_bins[i] - 1] = w[:-1]
                self.bias[i, : n_bins[i] - 1] = b[:-1]

    def get_max_n_bins(self) -> int:
        return self.weight.shape[-1]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.bias + self.weight * x[..., None]

        if x.shape[-1] > 1:
            first = mx.clip(x[..., :1], None, 1.0)
            middle = mx.clip(x[..., 1:-1], 0.0, 1.0)

            if self.single_bin_mask is None:
                last = mx.clip(x[..., -1:], 0.0, None)
            else:
                last = mx.where(
                    self.single_bin_mask[..., None],
                    x[..., -1:],
                    mx.clip(x[..., -1:], 0.0, None),
                )

            x = mx.concatenate([first, middle, last], axis=-1)

        return x


class _NLinear(nn.Module):
    """N separate linear layers for N feature embeddings."""

    def __init__(self, n: int, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = mx.zeros((n, in_features, out_features))
        if bias:
            self.bias = mx.zeros((n, out_features))
        else:
            self.bias = None

        # Initialize parameters
        scale = in_features**-0.5
        self.weight = mx.random.uniform(-scale, scale, self.weight.shape)
        if bias:
            self.bias = mx.random.uniform(-scale, scale, self.bias.shape)

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim != 3:
            raise ValueError(
                "Input must have shape (batch_size, n_features, d_embedding)"
            )

        # Transpose for matmul
        x = x.transpose(1, 0, 2)
        x = x @ self.weight
        x = x.transpose(1, 0, 2)

        if self.bias is not None:
            x = x + self.bias

        return x


class LinearEmbeddings(nn.Module):
    """Linear embeddings for continuous features."""

    def __init__(self, n_features: int, d_embedding: int):
        if n_features <= 0 or d_embedding <= 0:
            raise ValueError("n_features and d_embedding must be positive")

        super().__init__()
        scale = d_embedding**-0.5

        self.weight = mx.random.uniform(-scale, scale, (n_features, d_embedding))
        self.bias = mx.random.uniform(-scale, scale, (n_features, d_embedding))

    def __call__(self, x: mx.array) -> mx.array:
        return self.bias + self.weight * x[..., None]
