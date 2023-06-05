"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch

"""
An evaluation module for use with the OCP dataset and suite of tasks. It should
be possible to import this independently of the rest of the codebase, e.g:

```
from ocpmodels.modules import Evaluator

evaluator = Evaluator(task="is2re")
perf = evaluator.eval(prediction, target)
```

task: "s2ef", "is2rs", "is2re".

We specify a default set of metrics for each task, but should be easy to extend
to add more metrics. `evaluator.eval` takes as input two dictionaries, one for
predictions and another for targets to check against. It returns a dictionary
with the relevant metrics computed.
"""


class Evaluator:
    task_metrics = [
        "forces_mae",
        "forces_rmse",
        "energy_mae",
        "energy_mae_per_atom",
        "energy_rmse",
        "energy_rmse_per_atom",
    ]

    task_attributes = {
        "outputs": ["energy", "forces"],
    }

    def __init__(self):
        pass

    def eval(self, prediction, target):
        for attr in self.task_attributes["outputs"]:
            assert attr in prediction
            assert attr in target
            assert prediction[attr].shape == target[attr].shape

        # Fix some bugs
        prediction = {k: v.detach().cpu() for k, v in prediction.items()}
        target = {k: v.detach().cpu() for k, v in target.items()}
        metrics = {}
        for fn in self.task_metrics:
            res = eval(fn)(prediction, target).item()
            metrics[fn] = res

        return metrics


# For all below the prediciton and target take the same shape of
# forces being (T, n, 3) or (num_frames, num_atoms, 3)
# energy being (T, 1) or (num_frames, 1)
def forces_mae(prediction, target):
    """Mean absolute error for forces

    Don't normalize by number of atoms or by dimension (3)"""
    e = torch.abs(prediction["forces"] - target["forces"])
    return e.sum(dim=(2)).mean()


def forces_rmse(prediction, target):
    e = (prediction["forces"] - target["forces"]) ** 2
    return torch.sqrt(e.sum(dim=(2)).mean())


def energy_mae(prediction, target):
    """Mean absolute error for energy

    Don't normalize by number of atoms or by dimension (3)"""
    e = torch.abs(prediction["energy"] - target["energy"])
    return e.mean()


def energy_mae_per_atom(prediction, target):
    """Mean absolute error for energy per atom

    Normalize by number of atoms in addition to number of frames"""
    e = torch.abs(prediction["energy"] - target["energy"])
    num_atoms = prediction["forces"].shape[1]
    return e.mean() / num_atoms


def energy_rmse(prediction, target):
    e = (prediction["energy"] - target["energy"]) ** 2
    return torch.sqrt(e.mean())


def energy_rmse_per_atom(prediction, target):
    e = (prediction["energy"] - target["energy"]) ** 2
    num_atoms = prediction["forces"].shape[1]
    return torch.sqrt(e.mean()) / num_atoms
