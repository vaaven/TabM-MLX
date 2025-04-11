import itertools
from typing import Any, Literal

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import mlx.optimizers as optim

from embeddings import PiecewiseLinearEmbeddings

# ======================================================================================
# Initialization
# ======================================================================================


def init_rsqrt_uniform(shape, d: int):
    d_rsqrt = d**-0.5
    return mx.random.uniform(-d_rsqrt, d_rsqrt, shape=shape)


def init_random_signs(shape):
    return mx.where(mx.random.bernoulli(0.5, shape), 1, -1)


# ======================================================================================
# Modules
# ======================================================================================


class NLinear(nn.Module):
    def __init__(self, n: int, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = mx.random.normal(shape=(n, in_features, out_features))
        self.bias = mx.random.normal(shape=(n, out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        d = self.weight.shape[-2]
        self.weight = init_rsqrt_uniform(self.weight.shape, d)
        if self.bias is not None:
            self.bias = init_rsqrt_uniform(self.bias.shape, d)

    def __call__(self, x: mx.array) -> mx.array:
        x = x.transpose((1, 0, 2))
        x = x @ self.weight
        x = x.transpose((1, 0, 2))
        return x + self.bias if self.bias is not None else x


class OneHotEncoding0d(nn.Module):
    def __init__(self, cardinalities: list[int]):
        super().__init__()
        self.cardinalities = cardinalities

    def __call__(self, x: mx.array) -> mx.array:
        parts = []
        for i, c in enumerate(self.cardinalities):
            oh = mx.eye(c + 1)[x[..., i]][..., :-1]
            parts.append(oh)
        return mx.concatenate(parts, axis=-1)


class ScaleEnsemble(nn.Module):
    def __init__(
        self, k: int, d: int, *, init: Literal["ones", "normal", "random-signs"]
    ):
        super().__init__()
        self.weight = mx.random.normal(shape=(k, d))
        self.init = init
        self.reset_parameters()

    def reset_parameters(self):
        if self.init == "ones":
            self.weight = mx.ones_like(self.weight)
        elif self.init == "normal":
            self.weight = mx.random.normal(shape=self.weight.shape)
        elif self.init == "random-signs":
            self.weight = init_random_signs(self.weight.shape)

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.weight


class LinearEfficientEnsemble(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        k: int,
        ensemble_scaling_in: bool,
        ensemble_scaling_out: bool,
        ensemble_bias: bool,
        scaling_init: Literal["ones", "random-signs"]
    ):
        super().__init__()
        self.weight = mx.random.normal(shape=(out_features, in_features))
        self.r = (
            mx.random.normal(shape=(k, in_features)) if ensemble_scaling_in else None
        )
        self.s = (
            mx.random.normal(shape=(k, out_features)) if ensemble_scaling_out else None
        )
        self.bias = mx.random.normal(shape=(k, out_features)) if ensemble_bias else None
        self.scaling_init = scaling_init
        self.reset_parameters()

    def reset_parameters(self):
        self.weight = init_rsqrt_uniform(self.weight.shape, self.weight.shape[-1])
        init_fn = {"ones": mx.ones, "random-signs": init_random_signs}[
            self.scaling_init
        ]
        if self.s is not None:
            self.s = init_fn(self.s.shape)
        if self.bias is not None:
            self.bias = init_rsqrt_uniform(self.bias.shape, self.weight.shape[-1])

    def __call__(self, x: mx.array) -> mx.array:
        if self.r is not None:
            x *= self.r
        x = x @ self.weight.T
        if self.s is not None:
            x *= self.s
        return x + self.bias if self.bias is not None else x


class MLP(nn.Module):
    def __init__(
        self,
        *,
        d_in: int | None = None,
        d_out: int | None = None,
        n_blocks: int,
        d_block: int,
        dropout: float,
        activation: str = "relu"
    ):
        super().__init__()
        d_first = d_block if d_in is None else d_in
        self.blocks = [
            nn.Sequential(
                nn.Linear(d_first if i == 0 else d_block, d_block),
                getattr(nn, activation),
                nn.Dropout(dropout),
            )
            for i in range(n_blocks)
        ]
        self.output = nn.Linear(d_block, d_out) if d_out is not None else None

    def __call__(self, x: mx.array) -> mx.array:
        for block in self.blocks:
            x = block(x)
        return self.output(x) if self.output is not None else x


def make_efficient_ensemble(module: nn.Module, EnsembleLayer, **kwargs) -> None:
    """Replace linear layers with efficient ensembles of linear layers.

    NOTE
    In the paper, there are no experiments with networks with normalization layers.
    Perhaps, their trainable weights (the affine transformations) also need
    "ensemblification" as in the paper about "FiLM-Ensemble".
    Additional experiments are required to make conclusions.
    """
    if isinstance(module, nn.Sequential):
        for i in range(len(module.layers)):
            if isinstance(module.layers[i], nn.Linear):
                module.layers[i] = EnsembleLayer(
                    in_features=module.layers[i].weight.shape[1],
                    out_features=module.layers[i].weight.shape[0],
                    bias=module.layers[i].bias is not None,
                    **kwargs,
                )
    else:
        for name, submodule in module.named_modules():
            if not name:
                continue
            if isinstance(submodule, nn.Linear):
                submodule = EnsembleLayer(
                    in_features=submodule.weight.shape[1],
                    out_features=submodule.weight.shape[0],
                    bias=submodule.bias is not None,
                    **kwargs,
                )
            elif isinstance(submodule, nn.Module) or isinstance(
                submodule, nn.Sequential
            ):
                make_efficient_ensemble(submodule, EnsembleLayer, **kwargs)


# ======================================================================================
# Model
# ======================================================================================


class Model(nn.Module):
    def __init__(
        self,
        *,
        n_num_features: int,
        cat_cardinalities: list[int],
        n_classes: int | None,
        backbone: dict,
        num_embeddings: dict | None,
        arch_type: Literal[
            "plain",
            "tabm",
            "tabm-mini",
            "tabm-packed",
            "tabm-normal",
            "tabm-mini-normal",
        ],
        k: int | None = None,
        share_training_batches: bool = True
    ):
        super().__init__()
        # Numerical features processing
        self.num_module = None
        first_adapter_sections = []
        if n_num_features > 0:
            if num_embeddings:
                assert "d_embedding" in num_embeddings and "bins" in num_embeddings
                self.num_module = PiecewiseLinearEmbeddings(**num_embeddings)
                d_num = n_num_features * num_embeddings["d_embedding"]
                first_adapter_sections.extend(
                    [num_embeddings["d_embedding"]] * n_num_features
                )
            else:
                d_num = n_num_features
                first_adapter_sections.extend([1] * n_num_features)

        # Categorical features processing
        self.cat_module = (
            OneHotEncoding0d(cat_cardinalities) if cat_cardinalities else None
        )
        d_cat = sum(cat_cardinalities)
        first_adapter_sections.extend(cat_cardinalities)

        # Backbone initialization
        self.d_flat = d_num + d_cat
        self.backbone = MLP(d_in=self.d_flat, **backbone)
        self.minimal_ensemble_adapter = None

        # Ensemble initialization
        if arch_type != "plain":
            self._init_ensemble(arch_type, k, first_adapter_sections, num_embeddings)

        # Output layer
        self.output = self._init_output_layer(
            arch_type, backbone["d_block"], n_classes, k
        )

        self.arch_type = arch_type
        self.k = k
        self.share_training_batches = share_training_batches

    def _init_ensemble(self, arch_type, k, sections, num_embeddings):
        first_adapter_init = (
            None
            if arch_type == "tabm-packed"
            else (
                "normal"
                if arch_type in ("tabm-mini-normal", "tabm-normal")
                # For other arch_types, the initialization depends
                # on the presense of num_embeddings.
                else "random-signs" if num_embeddings is None else "normal"
            )
        )
        if arch_type in ("tabm", "tabm-normal"):
            assert first_adapter_init is not None
            make_efficient_ensemble(
                self.backbone,
                LinearEfficientEnsemble,
                k=k,
                ensemble_scaling_in=True,
                ensemble_scaling_out=True,
                ensemble_bias=True,
                scaling_init="ones",
            )
        elif arch_type in ("tabm-mini", "tabm-mini-normal"):
            self.minimal_ensemble_adapter = ScaleEnsemble(
                k,
                self.d_flat,
                init="random-signs" if num_embeddings is None else "normal",
            )

    def _init_output_layer(self, arch_type, d_block, n_classes, k):
        if arch_type == "plain":
            return nn.Linear(d_block, n_classes or 1)
        return NLinear(k, d_block, n_classes or 1)

    def __call__(
        self, x_num: mx.array | None = None, x_cat: mx.array | None = None
    ) -> mx.array:
        # Feature processing
        parts = []
        if x_num is not None:
            parts.append(x_num if self.num_module is None else self.num_module(x_num))
        if x_cat is not None:
            parts.append(self.cat_module(x_cat))
        x = mx.concatenate([p.reshape(p.shape[0], -1) for p in parts], axis=-1)

        # Ensemble processing
        if self.k is not None:
            if self.share_training_batches or not self.training:
                # (B, D) -> (B, K, D)
                x = mx.broadcast_to(x[:, None], (x.shape[0], self.k, x.shape[1]))
            else:
                # (B * K, D) -> (B, K, D)
                x = x.reshape(len(x) // self.k, self.k, x.shape[1])
            if self.minimal_ensemble_adapter:
                x = self.minimal_ensemble_adapter(x)

        # Backbone and output
        x = self.backbone(x)
        x = self.output(x)
        if self.k is None:
            x = x[:, None]
        return x
