import copy
import datetime
import logging
import time
from pathlib import Path
from pprint import pprint

import torch
from torch import nn
from torch_geometric.data import Batch

from ocpmodels.transfer_learning.common.utils import aggregate_metric
from ocpmodels.transfer_learning.models.distribution_regression import (
    EmbeddingKernel,
    EmbeddingRidgeRegression,
    GaussianKernel,
    LinearKernel,
    median_heuristic,
)

from ..loaders import BaseLoader
from .base import BaseTrainer


class MEKRRTrainer(BaseTrainer):
    def __init__(
        self,
        dataset_config,
        kernel_config,
        optimizer_config,
        logger_config,
        print_every=10,
        seed=None,
        cpu=False,
        name="MeanEmbeddingKRRtrainer",
        run_dir="checkpoints",
        is_debug=False,
    ):
        self.dataset_config = copy.deepcopy(dataset_config)  # Config for dataset
        self.kernel_config = kernel_config  # Config for kernel algorithm
        self.logger_config = copy.deepcopy(logger_config)  # Config for logger
        self.optimizer = copy.deepcopy(optimizer_config)  # Config for optimizer
        self.optimizer["energy_loss_coefficient"] = 1 - self.optimizer.get("force_loss_coefficient", 0.0)
        self.run_dir = run_dir
        self.path_run_dir = Path(self.run_dir)
        self.path_run_dir.mkdir(parents=True, exist_ok=True)
        self.cpu = cpu
        self.print_every = print_every  # Not used here
        self.seed = seed
        self.run_dir = run_dir
        self.timestamp_id = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        # Setup paths
        self.base_path = self.path_run_dir / self.timestamp_id
        self.checkpoint_dir = self.base_path / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir = self.base_path / "predictions"
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.is_debug = is_debug

        if torch.cuda.is_available() and not self.cpu:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            self.cpu = True

        # Load model config directly from pretrained checkpoint
        # and massage into right form
        self.model_config = torch.load(self.kernel_config["model_checkpoint_path"], map_location="cpu")["config"]
        name = self.model_config["model"]
        self.model_config = self.model_config["model_attributes"]
        self.model_config["regress_forces"] = True

        self.config = {
            "model": name,
            "model_attributes": self.model_config,
            "kernel": self.kernel_config,
            "logger": self.logger_config,
            "optim": self.optimizer,
            "name": name,
            "timestamp_id": self.timestamp_id,
            "dataset": self.dataset_config,
        }

        pprint(self.config)

        self.load()

    def load(self):
        self.load_seed_from_config()
        self.load_logger()
        self.load_datasets()
        self.load_normalizers()
        self._load_data_internal()
        self.load_metrics()
        self.load_model()

    def _load_data_internal(self):
        # Tranform the data into the a full batch
        for split in ["train", "val", "test"]:
            self.datasets[split] = Batch.from_data_list(self.datasets[split])

    def load_model(self):
        _config = copy.deepcopy(self.config)
        loader = BaseLoader(
            _config,
            representation=True,
            representation_kwargs=_config["kernel"]["representation_params"],
            regress_forces=_config["model_attributes"].pop("regress_forces"),
            seed=self.seed,
            cpu=True,  # Since we loaded the model onto the cpu to start
        )
        loader.load_checkpoint(self.kernel_config["model_checkpoint_path"], strict_load=False)
        self.model = loader.model.to(self.device)
        self.model.eval()
        logging.info("Loaded model from checkpoint successfully!")
        if self.logger is not None:
            self.logger.watch(self.model)

    def get_x_and_z_and_pos(self, split, pos_requires_grad=False):
        dataset = Batch.from_data_list(self.datasets[split][:])
        dataset.pos.requires_grad_(pos_requires_grad)
        X, _ = self.model(dataset.to(self.device))
        X = X.reshape(-1, self.config["dataset"][split]["num_atoms"], X.shape[-1])
        ZX = dataset.atomic_numbers.reshape(X.shape[0], self.config["dataset"][split]["num_atoms"], 1).to(self.device)
        return X, ZX, dataset.pos

    def train(self):
        start_time = time.time()
        # Set up data, taking care of normalization
        y = self.normalizers["target"].norm(self.datasets["train"].y.float().to(self.device)).reshape(-1, 1)
        # TODO: Implement gradient fitting
        # grad = self.normalizers["grad_target"].norm(self.datasets["train"].force.float().to(self.device))
        X, ZX, _ = self.get_x_and_z_and_pos("train")
        self.d = X.shape[-1]

        # Dispatch to kernel
        if self.config["kernel"].get("k0", "gaussian") == "gaussian":
            # Use median heuristic for Gaussian kernel.
            if (
                self.config["kernel"].get("k0", "gaussian") == "gaussian"
                and self.config["kernel"]["k0_params"].get("median_heuristic", True) is True
            ):
                with torch.no_grad():
                    self.k0_sigma = median_heuristic(X.reshape(-1, self.d), X.reshape(-1, self.d))
            else:
                self.k0_sigma = self.config["kernel"]["k0_params"].get("sigma", 1.0)
            self.k0 = GaussianKernel(self.k0_sigma)
        elif self.config["kernel"].get("k0", "linear") == "linear":
            self.k0 = LinearKernel()
        else:
            raise NotImplementedError

        if self.config["kernel"].get("k1", "linear") == "linear":
            self.k1 = EmbeddingKernel(
                self.k0,
                alpha=self.config["kernel"].get("alpha", 0.0),
                aggregation=self.config["kernel"].get("aggregation", "mean"),
            )
        else:
            raise NotImplementedError

        self.regressor = EmbeddingRidgeRegression(self.k1, **self.kernel_config["regressor_params"])
        assert (
            X.shape[0] == self.dataset_config["train"]["num_frames"]
            and X.shape[1] == self.dataset_config["train"]["num_atoms"]
        ), "X should have shape (num_frames, num_atoms, d)"
        self.regressor.fit(X, ZX, y)
        end_time = time.time()
        if self.logger is not None:
            self.logger.log({"train_time": end_time - start_time}, step=0, split="train")

    def validate(self, split="val"):
        start_time = time.time()
        dataset = self.datasets[split]
        num_atoms = self.dataset_config[split]["num_atoms"]

        predictions = self.predict(split)
        targets = {
            "energy": dataset.y.float().to(self.device),
            "forces": dataset.force.float().to(self.device).reshape(-1, num_atoms, 3),
        }
        metrics = self.evaluator.eval(predictions, targets)
        end_time = time.time()

        log_dict = {k: metrics[k] for k in metrics}
        log_dict.update({"epoch": 0})
        log_str = ["{}: {:.4f}".format(k, v) for k, v in log_dict.items()]
        logging.info(", ".join(log_str))

        if self.logger is not None:
            self.logger.log(
                log_dict,
                step=0,
                split=split,
            )
            self.logger.log({"eval_time": end_time - start_time}, step=0, split=split)

        return metrics

    def _forward(self, split):
        dataset = self.datasets[split]
        num_atoms = self.dataset_config[split]["num_atoms"]
        # forward pass
        X, ZX, pos = self.get_x_and_z_and_pos(split, pos_requires_grad=True)
        if self.config["model_attributes"].get("regress_forces", True):
            out_energy, out_grad = self.regressor.predict_y_and_grad(X, ZX, pos)
            out_forces = -out_grad.reshape(-1, 3)
        else:
            out_energy = self.regressor.predict(X, ZX)

        out_energy = out_energy.view(-1)
        # if out_energy.shape[-1] == 1:
        #     out_energy = out_energy.view(-1)

        # TODO: Don't hardcode float
        out = {
            "energy": out_energy.float(),
        }

        if self.config["model_attributes"].get("regress_forces", True):
            out["forces"] = out_forces.float()

        return out

    # Takes in a new data source and generates predictions on it.
    def predict(self, split):
        logging.info(f"Predicting {split}.")
        self.model.eval()
        predictions = {"energy": [], "forces": []}
        out = self._forward(split)
        # denorm
        out["energy"] = self.normalizers["target"].denorm(out["energy"])
        out["forces"] = self.normalizers["grad_target"].denorm(out["forces"])
        predictions["energy"] = out["energy"].detach()
        predictions["forces"] = out["forces"].detach().reshape(-1, self.dataset_config[split]["num_atoms"], 3)
        return predictions
