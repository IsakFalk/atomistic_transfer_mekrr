import copy
import errno
import logging
import os
import random

import numpy as np
import torch
import yaml
from torch.nn.parallel.distributed import DistributedDataParallel

from ocpmodels.common import distutils
from ocpmodels.common.data_parallel import OCPDataParallel
from ocpmodels.common.registry import registry


class BaseLoader:
    """Base class for model loaders.

    Load a model from a config file and potentially load the checkpoint state_dict
    """

    # Unused for now
    @property
    def _unwrapped_model(self):
        module = self.model
        while isinstance(module, (OCPDataParallel, DistributedDataParallel)):
            module = module.module
        return module

    def __init__(
        self,
        model,
        representation: bool = False,
        representation_kwargs: dict = {},
        regress_forces: bool = True,
        use_pbc: bool = True,
        seed=None,
        cpu=False,
        name="base_model_loader",
    ):
        self.name = name
        self.representation = representation
        self.representation_kwargs = representation_kwargs
        self.regress_forces = regress_forces
        self.use_pbc = use_pbc
        self.cpu = cpu  # TODO: Have not been tested with cuda but should work
        self.num_targets = 1  # NOTE: This is due to OCP code and should be fixed to 1

        # Don't mutate the original model dict
        model = copy.deepcopy(model)

        if torch.cuda.is_available() and not self.cpu:
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
            self.cpu = True

        # Due to how the internals of OCP work, we need to separate the name of the model class
        # and the attributes of the model class. This is done by the "model" and "model_attributes".
        # If the config is loaded from a checkpoint file, this allows us to recreate the correct model
        # and load the state_dict automatically
        self.config = {
            "model": model.pop("model"),
            "model_attributes": model["model_attributes"],
            "seed": seed,
        }
        self.config["model_attributes"]["use_pbc"] = self.use_pbc

        # Print the current config to stdout
        # print(yaml.dump(self.config, default_flow_style=False))
        self.load()

    def load(self):
        self.load_seed_from_config()
        self.load_model()

    def load_seed_from_config(self):
        """Set random seed from config file."""
        # https://pytorch.org/docs/stable/notes/randomness.html
        seed = self.config["seed"]
        if seed is None:
            return

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load_model(self):
        """Load the model from the config file."""

        # The OCP repo use a funny registry which allows them to load classes
        # using strings through a key-value store mapping strings to the correct
        # class object
        # This makes the registry available
        # Doesn't work for now
        from ocpmodels.common.utils import setup_imports

        setup_imports()

        # Build model
        if distutils.is_master():
            logging.info(f"Loading model: {self.config['model']}")
            if self.representation:
                logging.info("Model used for representation")

        # TODO: Says it's deprecated in the OCP code but it's required for now
        bond_feat_dim = None
        bond_feat_dim = self.config["model_attributes"].get("num_gaussians", 50)

        # Load the model class from the registry
        self.model = registry.get_model_class(self.config["model"])(
            num_atoms=None,
            bond_feat_dim=bond_feat_dim,
            num_targets=self.num_targets,
            representation=self.representation,
            regress_forces=self.regress_forces,
            **self.representation_kwargs,
            **self.config["model_attributes"],
        )

        if distutils.is_master():
            logging.info(f"Loaded {self.model.__class__.__name__} with " f"{self.model.num_params} parameters.")

    def load_checkpoint(self, checkpoint_path, strict_load=True):
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(errno.ENOENT, "Checkpoint file not found", checkpoint_path)

        logging.info(f"Loading checkpoint from: {checkpoint_path}")
        map_location = torch.device("cpu") if self.cpu else self.device
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        self.epoch = checkpoint.get("epoch", 0)
        self.step = checkpoint.get("step", 0)
        self.best_val_metric = checkpoint.get("best_val_metric", None)
        self.primary_metric = checkpoint.get("primary_metric", None)

        # Match the "module." count in the keys of model and checkpoint state_dict
        # DataParallel model has 1 "module.",  DistributedDataParallel has 2 "module."
        # Not using either of the above two would have no "module."

        ckpt_key_count = next(iter(checkpoint["state_dict"])).count("module")
        mod_key_count = next(iter(self.model.state_dict())).count("module")
        key_count_diff = mod_key_count - ckpt_key_count

        if key_count_diff > 0:
            new_dict = {key_count_diff * "module." + k: v for k, v in checkpoint["state_dict"].items()}
        elif key_count_diff < 0:
            new_dict = {k[len("module.") * abs(key_count_diff) :]: v for k, v in checkpoint["state_dict"].items()}
        else:
            new_dict = checkpoint["state_dict"]

        # NOTE: Their custom state_dict loader breaks for some unknown reason
        # related to the keys. This is due to the method ocpmodels.common.utils._report_incompat_keys
        # load_state_dict(self.model, new_dict, strict=strict_load)
        # Instead we mimic the checks they would do here taking care of errors
        incompat_keys = self.model.load_state_dict(new_dict, strict=strict_load)

        error_msgs = []
        if len(incompat_keys.unexpected_keys) > 0:
            error_msgs.insert(
                0,
                "Unexpected key(s) in state_dict: {}. ".format(
                    ", ".join('"{}"'.format(k) for k in incompat_keys.unexpected_keys)
                ),
            )
        if len(incompat_keys.missing_keys) > 0:
            error_msgs.insert(
                0,
                "Missing key(s) in state_dict: {}. ".format(
                    ", ".join('"{}"'.format(k) for k in incompat_keys.missing_keys)
                ),
            )

        if len(error_msgs) > 0:
            error_msg = "Error(s) in loading state_dict for {}:\n\t{}".format(
                self.model.__class__.__name__, "\n\t".join(error_msgs)
            )
            if strict_load:
                raise RuntimeError(error_msg)
            else:
                logging.warning(error_msg)
