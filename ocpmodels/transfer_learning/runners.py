from abc import ABC, abstractmethod

import numpy as np

from ocpmodels.transfer_learning.common.utils import torch_tensor_to_npy
from ocpmodels.transfer_learning.trainers.gap import GAPTrainer
from ocpmodels.transfer_learning.trainers.gnn import GNNTrainer
from ocpmodels.transfer_learning.trainers.mekrr import MEKRRTrainer
from ocpmodels.transfer_learning.trainers.mekrr_gnn import MEKRRGNNTrainer


class BaseRunner(ABC):
    def __init__(self, config, run_args):
        self.config = config
        self.run_args = run_args

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def run(self):
        pass


class MEKRRRunner(BaseRunner):
    def setup(self):
        self.trainer = MEKRRTrainer(
            self.config["dataset"],
            self.config["kernel"],
            self.config["optim"],
            self.config["logger"],
            print_every=self.run_args.print_every,
            seed=self.run_args.seed,
            cpu=self.run_args.cpu,
            name=self.config["logger"]["name"],
            run_dir=self.run_args.run_dir,
            is_debug=self.run_args.debug,
        )

    def run(self):
        if self.config["task"].get("train", True):
            self.trainer.train()
        for split in self.config["task"]["validate"]:
            self.trainer.validate(split=split)

        array_dict = {}
        for split in self.config["task"]["predict"]:
            out = self.trainer.predict(split)
            out = {k: torch_tensor_to_npy(v) for k, v in out.items()}
            for key, val in out.items():
                array_dict[f"{split}_{key}"] = val

        with open(self.trainer.predictions_dir / "predictions.npz", "wb") as f:
            np.savez(f, **array_dict)

        if not self.trainer.is_debug:
            self.trainer.logger.log_predictions(self.trainer.predictions_dir)


class GAPRunner(BaseRunner):
    def setup(self):
        self.trainer = GAPTrainer(
            self.config["dataset"],
            self.config["model"],
            self.config["logger"],
            print_every=self.run_args.print_every,
            seed=self.run_args.seed,
            cpu=self.run_args.cpu,
            name=self.config["logger"]["name"],
            run_dir=self.run_args.run_dir,
            is_debug=self.run_args.debug,
        )

    def run(self):
        if self.config["task"].get("train", True):
            self.trainer.train()
        for split in self.config["task"]["validate"]:
            self.trainer.validate(split=split)
        # Note that we predict twice, so there is some redundancy in computation here
        array_dict = {}
        for split in self.config["task"]["predict"]:
            out = self.trainer.predict(split)
            out = {k: torch_tensor_to_npy(v) for k, v in out.items()}
            for key, val in out.items():
                array_dict[f"{split}_{key}"] = val

        with open(self.trainer.predictions_dir / "predictions.npz", "wb") as f:
            np.savez(f, **array_dict)

        if not self.trainer.is_debug:
            self.trainer.logger.log_predictions(self.trainer.predictions_dir)


class GNNRunner(BaseRunner):
    def setup(self):
        self.trainer = GNNTrainer(
            self.config["task"],
            self.config["model"],
            self.config["dataset"],
            self.config["optim"],
            self.config["logger"],
            print_every=self.run_args.print_every,
            seed=self.run_args.seed,
            cpu=self.run_args.cpu,
            name=self.config["logger"]["name"],
            run_dir=self.run_args.run_dir,
            is_debug=self.run_args.debug,
            hide_eval_progressbar=self.config.get("hide_eval_progressbar", False),
        )
        # TODO: add checkpoint resuming

    def run(self):
        if self.config["task"].get("train", True):
            self.trainer.train(
                disable_eval_tqdm=self.config.get("hide_eval_progressbar", False),
            )


class MEKRRGNNRunner(BaseRunner):
    def setup(self):
        self.trainer = MEKRRGNNTrainer(
            self.config["task"],
            self.config["model"],
            self.config["dataset"],
            self.config["optim"],
            self.config["logger"],
            print_every=self.run_args.print_every,
            seed=self.run_args.seed,
            cpu=self.run_args.cpu,
            name=self.config["logger"]["name"],
            run_dir=self.run_args.run_dir,
            is_debug=self.run_args.debug,
            hide_eval_progressbar=self.config.get("hide_eval_progressbar", False),
        )
        # TODO: add checkpoint resuming

    def run(self):
        if self.config["task"].get("train", True):
            self.trainer.train(
                disable_eval_tqdm=self.config.get("hide_eval_progressbar", False),
            )


class FTGNNRunner(BaseRunner):
    def setup(self):
        self.trainer = GNNTrainer(
            self.config["task"],
            self.config["model"],
            self.config["dataset"],
            self.config["optim"],
            self.config["logger"],
            print_every=self.run_args.print_every,
            seed=self.run_args.seed,
            cpu=self.run_args.cpu,
            name=self.config["logger"]["name"],
            run_dir=self.run_args.run_dir,
            is_debug=self.run_args.debug,
            hide_eval_progressbar=self.config.get("hide_eval_progressbar", False),
            fine_tune=True,
        )
        # TODO: add checkpoint resuming

    def run(self):
        if self.config["task"].get("train", True):
            self.trainer.train(
                disable_eval_tqdm=self.config.get("hide_eval_progressbar", False),
            )
        for split in self.config["task"]["validate"]:
            self.trainer.validate(split=split, final=True)
        for split in self.config["task"]["predict"]:
            self.trainer.predict(split=split)
