import copy
import datetime
import logging
import math
import re
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from ocpmodels.common.utils import save_checkpoint
from ocpmodels.modules.scheduler import LRScheduler
from ocpmodels.transfer_learning.common.utils import (
    aggregate_metric,
    torch_tensor_to_npy,
)
from ocpmodels.transfer_learning.models.distribution_regression import (
    median_heuristic,
)

from ..loaders import BaseLoader
from .base import BaseTrainer

### First create random feature kernel
# class RFFFeatureMap(torch.nn.Module):
#     def __init__(self, D, d, sigma):
#         super(RFFFeatureMap, self).__init__()
#         self.D = D
#         self.d = d
#         self.sigma = sigma
#         self.w = self._sample_w()

#     def _sample_w(self):
#         return torch.randn(self.D, self.d)

#     def forward(self, x):
#         q = torch.matmul(x, self.w.T)
#         zcos = torch.cos(q / self.sigma)
#         zsin = torch.sin(q / self.sigma)
#         z = math.sqrt(2 / self.D) * torch.cat([zcos, zsin], dim=-1)
#         return z


class RFFFeatureMap(torch.nn.Module):
    def __init__(self, D, d, sigma):
        super(RFFFeatureMap, self).__init__()
        self.D = D
        self.d = d
        self.sigma = sigma
        self.w, self.b = self._sample_w_and_b()

    def _sample_w_and_b(self):
        # Parameter so we can move module to device
        w = nn.parameter.Parameter(torch.randn(self.D, self.d))
        b = nn.parameter.Parameter(torch.rand(self.D, 1) * np.pi * 2)
        return w, b

    def forward(self, x):
        q = torch.matmul(x, self.w.T) / self.sigma + self.b.T
        z = math.sqrt(2 / self.D) * torch.cos(q)
        return z


class MERFFGNN(torch.nn.Module):
    def __init__(self, D, d, sigma, model, regress_forces=True):
        super(MERFFGNN, self).__init__()
        self.D = D
        self.d = d
        self.sigma = sigma
        self.gnn = model
        self.feature_map = RFFFeatureMap(D, d, sigma)
        self.gnn.regress_forces = False
        self.w = nn.Linear(D, 1, bias=False)
        self.regress_forces = regress_forces

    def forward(self, data):
        if self.regress_forces:
            data.pos.requires_grad_(True)

        frames = len(torch.unique(data.batch))
        num_atoms = data.natoms[0]
        h = self.gnn(data)
        d = h.shape[-1]
        mu = self.feature_map(h).reshape(frames, num_atoms, -1).mean(1)
        energy = self.w(mu)

        if self.regress_forces:
            forces = -1 * (
                torch.autograd.grad(
                    energy,
                    data.pos,
                    grad_outputs=torch.ones_like(energy),
                    create_graph=True,
                )[0]
            )
            return energy, forces

        else:
            return energy


class MEKRRGNNTrainer(BaseTrainer):
    def __init__(
        self,
        task_config,
        model_config,
        dataset_config,
        optimizer_config,
        logger_config,
        print_every=30,
        seed=None,
        cpu=False,
        name="trainer",
        run_dir="checkpoints",
        is_debug=False,
        hide_eval_progressbar=False,
    ):
        self.task_config = copy.deepcopy(task_config)
        self.model_config = copy.deepcopy(model_config)  # Config for model
        self.dataset_config = copy.deepcopy(dataset_config)  # Config for dataset
        self.optimizer = copy.deepcopy(optimizer_config)  # Config for optimizer
        self.logger_config = copy.deepcopy(logger_config)  # Config for logger
        self.optimizer["energy_loss_coefficient"] = 1 - self.optimizer.get("force_loss_coefficient", 0.0)
        self.model_config["regress_forces"] = True
        self.cpu = cpu
        self.print_every = print_every
        self.seed = seed
        self.run_dir = run_dir
        self.path_run_dir = Path(self.run_dir)
        self.timestamp_id = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        # Setup paths
        self.base_path = self.path_run_dir / self.timestamp_id
        self.checkpoint_dir = self.base_path / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir = self.base_path / "predictions"
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.is_debug = is_debug
        self.hide_eval_progressbar = hide_eval_progressbar
        self.epoch = 0
        self.step = 0
        if torch.cuda.is_available() and not self.cpu:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            self.cpu = True

        # Load model config directly from pretrained checkpoint
        # and massage into right form
        self.D = int(self.model_config.pop("D"))
        self.lmbda = float(self.model_config.pop("lmbda"))
        self.representation_params = self.model_config.pop("representation_params")
        # Load the config from the path
        self.model_checkpoint_path = self.model_config["model_checkpoint_path"]
        self.model_config = torch.load(self.model_checkpoint_path, map_location=self.device)["config"]
        name = self.model_config["model"]
        self.model_config = self.model_config["model_attributes"]
        self.model_config["regress_forces"] = True

        self.config = {
            "model": name,
            "model_attributes": self.model_config,
            "optim": self.optimizer,
            "logger": self.logger_config,
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
        self.load_loss()
        self.load_optimizer()
        self.load_extras()

    def _load_data_internal(self):
        self.loaders = {}
        for split in ["train", "val", "test"]:
            self.loaders[split] = self.get_dataloader(self.datasets[split])

    def get_dataloader(self, list_of_data):
        loader = DataLoader(
            list_of_data,
            batch_size=self.config["optim"]["batch_size"],
            num_workers=self.config["optim"]["num_workers"],
            pin_memory=True,
        )
        return loader

    def load_model(self):
        _config = copy.deepcopy(self.config)
        regress_forces = _config["model_attributes"].pop("regress_forces")
        loader = BaseLoader(
            _config,
            representation=True,
            representation_kwargs=self.representation_params,
            regress_forces=False,
            seed=self.seed,
            cpu=self.cpu,
        )
        loader.load_checkpoint(self.model_checkpoint_path, strict_load=False)
        self.base_model = loader.model.to(self.device)
        self.base_model.eval()
        # Get sigma using median heuristic
        with torch.no_grad():
            h = self.base_model(Batch.from_data_list(self.datasets["train"]).to(self.device)).cpu()
            self.d = h.shape[-1]
            self.sigma = median_heuristic(h.reshape(-1, self.d), h.reshape(-1, self.d))
            del h

        self.model = MERFFGNN(self.D, self.d, self.sigma, self.base_model, regress_forces).to(self.device)
        # Freeze all layers except for last one
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.w.weight.requires_grad = True

        if self.logger is not None:
            self.logger.watch(self.model)

    def load_loss(self):
        self.loss_fn = {}
        energy_train_objective = self.task_config["train_objective"]["energy"]
        if energy_train_objective == "mse":
            self.loss_fn["energy"] = nn.MSELoss()
        elif energy_train_objective == "mae":
            self.loss_fn["energy"] = nn.L1Loss()

        forces_train_objective = self.task_config["train_objective"]["forces"]
        if forces_train_objective == "mse":
            self.loss_fn["forces"] = nn.MSELoss()
        elif forces_train_objective == "mae":
            self.loss_fn["forces"] = nn.L1Loss()

    def load_optimizer(self):
        optimizer = self.config["optim"].get("optimizer", "AdamW")
        optimizer = getattr(optim, optimizer)

        # if self.config["optim"].get("weight_decay", 0) > 0:
        #     # Do not regularize bias etc.
        #     params_decay = []
        #     params_no_decay = []
        #     for name, param in self.model.named_parameters():
        #         if param.requires_grad:
        #             if "embedding" in name:
        #                 params_no_decay += [param]
        #             elif "frequencies" in name:
        #                 params_no_decay += [param]
        #             elif "bias" in name:
        #                 params_no_decay += [param]
        #             else:
        #                 params_decay += [param]

        #     self.optimizer = optimizer(
        #         [
        #             {"params": params_no_decay, "weight_decay": 0},
        #             {
        #                 "params": params_decay,
        #                 "weight_decay": self.config["optim"]["weight_decay"],
        #             },
        #         ],
        #         lr=self.config["optim"]["lr_initial"],
        #         **self.config["optim"].get("optimizer_params", {}),
        #     )
        # else:
        self.optimizer = optimizer(
            params=[p for p in self.model.parameters() if p.requires_grad],
            lr=self.config["optim"]["lr_initial"],
            **self.config["optim"].get("optimizer_params", {}),
        )

    def load_extras(self):
        self.scheduler = LRScheduler(self.optimizer, self.config["optim"])  # for now no decay

    def update_best(
        self,
        primary_metric,
        val_metrics,
        disable_eval_tqdm=True,
    ):
        current_metric = val_metrics[primary_metric]
        if current_metric < self.best_val_metric:
            self.best_val_metric = current_metric
            self.save(
                metrics=val_metrics,
                checkpoint_file="best_checkpoint.pt",
                training_state=False,
            )
            # Log best model
            if self.logger is not None:
                self.logger.save_model(f"{self.run_dir}/checkpoints/best_checkpoint.pt")

    def train(self, disable_eval_tqdm=False):
        # ensure_fitted(self._unwrapped_model, warn=True)
        eval_every = self.config["optim"].get("eval_every", len(self.loaders["train"]))
        checkpoint_every = self.config["optim"].get("checkpoint_every", eval_every)
        primary_metric = self.task_config["primary_val_metric"]
        self.best_val_metric = np.inf

        self.loss_metrics = {"energy_loss": [], "forces_loss": []}
        self.step = 0
        for epoch in range(self.config["optim"]["max_epochs"]):
            self.epoch = epoch
            for batch in self.loaders["train"]:
                self.model.train()
                # Forward, loss, backward.
                out = self._forward(batch)
                loss, losses = self._compute_loss(out, batch)
                self._backward(
                    loss + self.lmbda * (self.model.w.weight**2).sum()
                )  # NB: we do KRR with lmbda ||w||^2 regularizer
                self.loss_metrics["energy_loss"].append(aggregate_metric(losses["energy"].item()))
                self.loss_metrics["forces_loss"].append(aggregate_metric(losses["forces"].item()))

                # Log loss metrics.
                log_dict = {k: aggregate_metric(self.loss_metrics[k]) for k in self.loss_metrics}
                log_dict.update(
                    {
                        "lr": self.scheduler.get_lr(),
                        "epoch": self.epoch,
                        "step": self.step,
                    }
                )
                if self.step % self.print_every == 0:
                    log_str = ["{}: {:.2e}".format(k, v) for k, v in log_dict.items()]
                    logging.info(", ".join(log_str))  # TODO: In future, this should output metrics, such as mae
                    self.loss_metrics = {"energy_loss": [], "forces_loss": []}

                if self.logger is not None:
                    self.logger.log(
                        log_dict,
                        step=self.step,
                        split="train",
                    )

                if checkpoint_every != -1 and self.step % checkpoint_every == 0:
                    self.save(checkpoint_file="checkpoint.pt", training_state=True)

                # Evaluate on val set every `eval_every` iterations.
                if self.step % eval_every == 0:
                    if self.loaders["val"] is not None:
                        val_metrics = self.validate(split="val", disable_tqdm=disable_eval_tqdm, final=False)
                        self.update_best(
                            primary_metric,
                            val_metrics,
                            disable_eval_tqdm=disable_eval_tqdm,
                        )

                # if self.scheduler.scheduler_type == "ReduceLROnPlateau"
                #     if self.step % eval_every == 0:
                #         self.scheduler.step(
                #             metrics=val_metrics[primary_metric]["metric"],
                #         )
                # else:
                #     self.scheduler.step()
                self.scheduler.step()
                self.step += 1

            torch.cuda.empty_cache()

            if checkpoint_every == -1:
                self.save(checkpoint_file="checkpoint.pt", training_state=True)

        # Retrieve best model
        self.model.load_state_dict(torch.load(self.checkpoint_dir / "best_checkpoint.pt")["state_dict"])

        # Save best performance
        for split in self.task_config["validate"]:
            self.validate(split=split, disable_tqdm=self.hide_eval_progressbar, final=True)

        # Save best predictions
        if "predict" in self.task_config:
            array_dict = {}
            for split in self.task_config["predict"]:
                out = self.predict(split, disable_tqdm=self.hide_eval_progressbar)
                out = {k: torch_tensor_to_npy(v) for k, v in out.items()}
                for key, val in out.items():
                    array_dict[f"{split}_{key}"] = val
            with open(self.predictions_dir / "predictions.npz", "wb") as f:
                np.savez(f, **array_dict)

            if not self.is_debug:
                self.logger.log_predictions(self.predictions_dir)

    # @torch.no_grad()
    def validate(self, split="val", disable_tqdm=False, final=False):
        self.model.eval()
        loader = self.loaders[split]

        # Get predictions and targets over the full 'split' dataset
        predictions = {"energy": [], "forces": []}
        targets = {"energy": [], "forces": []}
        for i, batch in tqdm(
            enumerate(loader),
            total=len(loader),
            disable=disable_tqdm,
        ):
            out = self._forward(batch)
            # denorm
            out["energy"] = self.normalizers["target"].denorm(out["energy"])
            out["forces"] = self.normalizers["grad_target"].denorm(out["forces"])
            predictions["energy"].extend(out["energy"].detach())
            predictions["forces"].extend(out["forces"].detach().reshape(-1, self.dataset_config[split]["num_atoms"], 3))
            targets["energy"].extend(batch.y.detach())
            targets["forces"].extend(batch.force.detach().reshape(-1, self.dataset_config[split]["num_atoms"], 3))
        predictions["energy"] = torch.stack(predictions["energy"])
        predictions["forces"] = torch.stack(predictions["forces"])
        targets["energy"] = torch.stack(targets["energy"])
        targets["forces"] = torch.stack(targets["forces"])

        metrics = self.evaluator.eval(predictions, targets)

        if not final:
            # If we are training, we use this to get a nice plot in wandb
            log_dict = {"loss_" + k: metrics[k] for k in metrics}
        else:
            # For this we will have the same name so we can compare across methods
            log_dict = {k: metrics[k] for k in metrics}
        log_dict.update({"epoch": self.epoch})
        log_str = ["{}: {:.4f}".format(k, v) for k, v in log_dict.items()]
        logging.info(", ".join(log_str))

        if self.logger is not None:
            self.logger.log(
                log_dict,
                step=self.step,
                split=split,
            )

        return metrics

    def _forward(self, batch):
        batch.to(self.device)
        # forward pass.
        if self.config["model_attributes"].get("regress_forces", True):
            out_energy, out_forces = self.model(batch)
        else:
            out_energy = self.model(batch)

        if out_energy.shape[-1] == 1:
            out_energy = out_energy.view(-1)

        # TODO: Don't hardcode float
        out = {
            "energy": out_energy.float(),
        }

        if self.config["model_attributes"].get("regress_forces", True):
            out["forces"] = out_forces.float()

        return out

    def _compute_loss(self, out, batch_list_or_batch):
        # NOTE: Removed some additional things we probably want
        if not isinstance(batch_list_or_batch, list):
            batch_list = [batch_list_or_batch]
        else:
            batch_list = batch_list_or_batch

        losses = dict()

        # Energy loss.
        energy_target = torch.cat([batch.y.to(self.device) for batch in batch_list], dim=0).float()
        energy_target = self.normalizers["target"].norm(energy_target)
        losses["energy"] = self.loss_fn["energy"](out["energy"], energy_target)
        # Force loss.
        if self.config["model_attributes"].get("regress_forces", True):
            force_target = torch.cat([batch.force.to(self.device) for batch in batch_list], dim=0).float()
            force_target = self.normalizers["grad_target"].norm(force_target)
            losses["forces"] = self.loss_fn["forces"](out["forces"], force_target)

        # Sanity check to make sure the compute graph is correct.
        for lc in losses.values():
            assert hasattr(lc, "grad_fn")

        loss = (
            self.config["optim"].get("energy_loss_coefficient") * losses["energy"]
            + self.config["optim"].get("force_loss_coefficient") * losses["forces"]
        )
        return loss, losses

    def _backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # Takes in a new data source and generates predictions on it.
    #@torch.no_grad()
    def predict(
        self,
        split,
        disable_tqdm=False,
    ):
        loader = self.loaders[split]
        logging.info("Predicting.")
        assert isinstance(
            loader,
            (
                torch.utils.data.dataloader.DataLoader,
                torch_geometric.data.Batch,
                torch_geometric.loader.dataloader.DataLoader,
            ),
        )

        if isinstance(loader, torch_geometric.data.Batch):
            loader = [[loader]]

        self.model.eval()
        predictions = {"energy": [], "forces": []}
        for i, batch in tqdm(
            enumerate(loader),
            total=len(loader),
            disable=disable_tqdm,
        ):
            out = self._forward(batch)
            # denorm
            out["energy"] = self.normalizers["target"].denorm(out["energy"])
            out["forces"] = self.normalizers["grad_target"].denorm(out["forces"])
            predictions["energy"].extend(out["energy"].detach())
            predictions["forces"].extend(out["forces"].detach().reshape(-1, self.dataset_config[split]["num_atoms"], 3))

        predictions["energy"] = torch.stack(predictions["energy"])
        predictions["forces"] = torch.stack(predictions["forces"])

        return predictions

    def save(
        self,
        metrics=None,
        checkpoint_file="checkpoint.pt",
        training_state=True,
    ):
        if training_state:
            config = {
                "epoch": self.epoch,
                "step": self.step,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.scheduler.state_dict() if self.scheduler.scheduler_type != "Null" else None,
                "normalizers": {key: value.state_dict() for key, value in self.normalizers.items()},
                "config": self.config,
                "val_metrics": metrics,
                # "ema": self.ema.state_dict() if self.ema else None,
                # "amp": self.scaler.state_dict()
                # if self.scaler
                # else None,
                "best_val_metric": self.best_val_metric,
                # "primary_metric": self.config["task"].get(
                #     "primary_metric",
                #     self.evaluator.task_primary_metric[self.name],
                # ),
            }
            save_checkpoint(
                config,
                checkpoint_dir=str(self.checkpoint_dir),
                checkpoint_file=checkpoint_file,
            )
        else:
            config = {
                "state_dict": self.model.state_dict(),
                "normalizers": {key: value.state_dict() for key, value in self.normalizers.items()},
                "config": self.config,
                "val_metrics": metrics,
                # "amp": self.scaler.state_dict()
                # if self.scaler
                # else None,
            }
            ckpt_path = save_checkpoint(
                config,
                checkpoint_dir=str(self.checkpoint_dir),
                checkpoint_file=checkpoint_file,
            )
            return ckpt_path
        return None
