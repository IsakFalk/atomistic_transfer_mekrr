import random

import numpy as np
import torch

from ocpmodels.modules.normalizer import Normalizer
from ocpmodels.transfer_learning.common.logger import WandBLogger
from ocpmodels.transfer_learning.common.utils import (
    ATOMS_TO_GRAPH_KWARGS,
    load_xyz_to_pyg_data,
)
from ocpmodels.transfer_learning.modules.evaluator import Evaluator


# TODO: Move common methods here especially metric calculations
# Should decouple load_losses from metrics
class BaseTrainer:
    def load_seed_from_config(self):
        # https://pytorch.org/docs/stable/notes/randomness.html
        if self.seed is None:
            return

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load_logger(self):
        self.logger = None
        if not self.is_debug:
            self.logger = WandBLogger(self.config)

    def _compute_metrics(self):
        # TODO: Move this to a common method
        pass

    def load_datasets(self):
        self.datasets = {}
        for split in ["train", "val", "test"]:
            _, dataset, num_frames, num_atoms = load_xyz_to_pyg_data(
                self.dataset_config[split]["src"], ATOMS_TO_GRAPH_KWARGS[self.config.get("model", "schnet")]
            )
            self.datasets[split] = dataset
            self.config["dataset"][split]["num_frames"] = num_frames
            self.config["dataset"][split]["num_atoms"] = num_atoms

    def load_normalizers(self):
        self.normalizer = self.config["dataset"]["train"]
        self.normalizers = {}
        if self.normalizer.get("normalize_labels", True):
            if "target_mean" in self.normalizer:
                self.normalizers["target"] = Normalizer(
                    mean=self.normalizer["target_mean"],
                    std=self.normalizer["target_std"],
                    device=self.device,
                )
            else:
                y = torch.tensor(np.array([data.y for data in self.datasets["train"]])).to(self.device)
                self.normalizers["target"] = Normalizer(
                    mean=y.mean().float(),
                    std=y.std().float(),
                    device=self.device,
                )
            if "grad_target_mean" in self.normalizer:
                self.normalizers["grad_target"] = Normalizer(
                    mean=self.normalizer["grad_target_mean"],
                    std=self.normalizer["grad_target_std"],
                    device=self.device,
                )
            else:
                forces = torch.cat([data.force.clone().detach() for data in self.datasets["train"]], dim=0)
                self.normalizers["grad_target"] = Normalizer(
                    mean=0.0,
                    std=forces.std().float(),
                    device=self.device,
                )
                self.normalizers["grad_target"].mean.fill_(0)
        else:
            self.normalizers["target"] = Normalizer(
                mean=0.0,
                std=1.0,
                device=self.device,
            )
            self.normalizers["grad_target"] = Normalizer(
                mean=0.0,
                std=1.0,
                device=self.device,
            )

    def load_metrics(self):
        self.evaluator = Evaluator()

    def _load_dataset_internal(self):
        pass
        # Here we will make it into the internal version

    def compute_final_metrics(self, split):
        """Compute final metrics for all metrics defined"""
        pass
