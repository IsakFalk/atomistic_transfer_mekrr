"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Optional

import torch
from torch_scatter import scatter
from torch_scatter.utils import broadcast

from ocpmodels.common import gp_utils
from ocpmodels.modules.scaling import ScaleFactor

from ..initializers import he_orthogonal_init
from .base_layers import Dense, ResidualLayer


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Clone of torch_scatter.scatter_sum but without in-place operations
    """
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1

        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return torch.scatter_add(out, dim, index, src)
    else:
        return out.scatter_add(dim, index, src)


class AtomUpdateBlock(torch.nn.Module):
    """
    Aggregate the message embeddings of the atoms

    Parameters
    ----------
        emb_size_atom: int
            Embedding size of the atoms.
        emb_size_atom: int
            Embedding size of the edges.
        nHidden: int
            Number of residual blocks.
        activation: callable/str
            Name of the activation function to use in the dense layers.
    """

    def __init__(
        self,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_rbf: int,
        nHidden: int,
        activation=None,
        name: str = "atom_update",
    ):
        super().__init__()
        self.name = name

        self.dense_rbf = Dense(emb_size_rbf, emb_size_edge, activation=None, bias=False)
        self.scale_sum = ScaleFactor(name + "_sum")

        self.layers = self.get_mlp(emb_size_edge, emb_size_atom, nHidden, activation)

    def get_mlp(self, units_in, units, nHidden, activation):
        dense1 = Dense(units_in, units, activation=activation, bias=False)
        mlp = [dense1]
        res = [ResidualLayer(units, nLayers=2, activation=activation) for i in range(nHidden)]
        mlp = mlp + res
        return torch.nn.ModuleList(mlp)

    def forward(self, nAtoms, m, rbf, id_j):
        """
        Returns
        -------
            h: torch.Tensor, shape=(nAtoms, emb_size_atom)
                Atom embedding.
        """
        mlp_rbf = self.dense_rbf(rbf)  # (nEdges, emb_size_edge)
        x = m * mlp_rbf

        # Graph Parallel: Local node aggregation
        x2 = scatter(x, id_j, dim=0, dim_size=nAtoms, reduce="sum")

        # Graph Parallel: Global node aggregation
        x2 = gp_utils.reduce_from_model_parallel_region(x2)
        x2 = gp_utils.scatter_to_model_parallel_region(x2, dim=0)

        # (nAtoms, emb_size_edge)
        x = self.scale_sum(x2, ref=m)

        for layer in self.layers:
            x = layer(x)  # (nAtoms, emb_size_atom)

        return x


class OutputBlock(AtomUpdateBlock):
    """
    Combines the atom update block and subsequent final dense layer.

    Parameters
    ----------
        emb_size_atom: int
            Embedding size of the atoms.
        emb_size_atom: int
            Embedding size of the edges.
        nHidden: int
            Number of residual blocks.
        num_targets: int
            Number of targets.
        activation: str
            Name of the activation function to use in the dense layers except for the final dense layer.
        direct_forces: bool
            If true directly predict forces without taking the gradient of the energy potential.
        output_init: int
            Kernel initializer of the final dense layer.
    """

    def __init__(
        self,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_rbf: int,
        nHidden: int,
        num_targets: int,
        activation=None,
        direct_forces=True,
        output_init="HeOrthogonal",
        name: str = "output",
        **kwargs,
    ):
        super().__init__(
            name=name,
            emb_size_atom=emb_size_atom,
            emb_size_edge=emb_size_edge,
            emb_size_rbf=emb_size_rbf,
            nHidden=nHidden,
            activation=activation,
            **kwargs,
        )

        assert isinstance(output_init, str)
        self.output_init = output_init.lower()
        self.direct_forces = direct_forces

        self.seq_energy = self.layers  # inherited from parent class
        self.out_energy = Dense(emb_size_atom, num_targets, bias=False, activation=None)

        if self.direct_forces:
            self.scale_rbf_F = ScaleFactor(name + "_had")
            self.seq_forces = self.get_mlp(emb_size_edge, emb_size_edge, nHidden, activation)
            self.out_forces = Dense(emb_size_edge, num_targets, bias=False, activation=None)
            self.dense_rbf_F = Dense(emb_size_rbf, emb_size_edge, activation=None, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        if self.output_init == "heorthogonal":
            self.out_energy.reset_parameters(he_orthogonal_init)
            if self.direct_forces:
                self.out_forces.reset_parameters(he_orthogonal_init)
        elif self.output_init == "zeros":
            self.out_energy.reset_parameters(torch.nn.init.zeros_)
            if self.direct_forces:
                self.out_forces.reset_parameters(torch.nn.init.zeros_)
        else:
            raise UserWarning(f"Unknown output_init: {self.output_init}")

    def forward(self, nAtoms, m, rbf, id_j):
        """
        Returns
        -------
            (E, F): tuple
            - E: torch.Tensor, shape=(nAtoms, num_targets)
            - F: torch.Tensor, shape=(nEdges, num_targets)
            Energy and force prediction
        """

        # -------------------------------------- Energy Prediction -------------------------------------- #
        rbf_emb_E = self.dense_rbf(rbf)  # (nEdges, emb_size_edge)
        x = m * rbf_emb_E

        # Graph Parallel: Local Node aggregation
        x_E = scatter(x, id_j, dim=0, dim_size=nAtoms, reduce="sum")
        # Graph Parallel: Global Node aggregation
        x_E = gp_utils.reduce_from_model_parallel_region(x_E)
        x_E = gp_utils.scatter_to_model_parallel_region(x_E, dim=0)

        # (nAtoms, emb_size_edge)
        x_E = self.scale_sum(x_E, ref=m)

        for layer in self.seq_energy:
            x_E = layer(x_E)  # (nAtoms, emb_size_atom)

        x_E = self.out_energy(x_E)  # (nAtoms, num_targets)

        # --------------------------------------- Force Prediction -------------------------------------- #
        if self.direct_forces:
            x_F = m
            for i, layer in enumerate(self.seq_forces):
                x_F = layer(x_F)  # (nEdges, emb_size_edge)

            rbf_emb_F = self.dense_rbf_F(rbf)  # (nEdges, emb_size_edge)
            x_F_rbf = x_F * rbf_emb_F
            x_F = self.scale_rbf_F(x_F_rbf, ref=x_F)

            x_F = self.out_forces(x_F)  # (nEdges, num_targets)
        else:
            x_F = 0
        # ----------------------------------------------------------------------------------------------- #

        return x_E, x_F
