import copy
from pathlib import Path

import ase.io
import torch
import yaml
from ase import Atoms
from torch_geometric.data import Batch

from ocpmodels.preprocessing import AtomsToGraphs

# Keyword arguments for converting ASE Atoms objects to PyTorch Geometric Batch objects per model
ATOMS_TO_GRAPH_KWARGS = {
    "schnet": {
        "max_neigh": 50,
        "radius": 6,
        "r_energy": True,
        "r_forces": True,
        "r_distances": False,
        "r_edges": True,
        "r_fixed": True,
    }
}


def torch_detach_maybe(x):
    """TODO: Maybe not correct. But should be okay."""
    if isinstance(x, list):
        return [torch_detach_maybe(y) for y in x]
    if isinstance(x, torch.Tensor):
        return x.clone().detach()
    else:
        return torch.tensor(x)


def aggregate_metric(metric):
    metric = torch.tensor(torch_detach_maybe(metric))
    return metric.mean().item()


def get_config(path):
    with open(path, "r") as f:
        original_config = yaml.safe_load(f)
        config = copy.deepcopy(original_config)
    return config


def load_xyz_to_pyg_batch(path: Path, atoms_to_graph_kwargs: dict) -> tuple[Atoms, Batch, int, int]:
    """
    Load XYZ data from a given path using ASE and convert it into a PyTorch Geometric Batch object.

    Args:
        path (Path): Path to the XYZ data file.
        **atoms_to_graph_kwargs: Optional keyword arguments for AtomsToGraphs class.

    Returns:
        Tuple consisting of raw_data, data_batch, num_frames, and num_atoms.
        raw_data (Atoms): Raw data loaded from the file using ASE.
        data_batch (Batch): Batch object containing converted data for all frames.
        num_frames (int): Number of frames in the loaded XYZ data file.
        num_atoms (int): Number of atoms in each frame.
    """
    raw_data = ase.io.read(path, index=":")
    num_frames = len(raw_data)
    a2g = AtomsToGraphs(
        **atoms_to_graph_kwargs,
    )
    data_object = a2g.convert_all(raw_data, disable_tqdm=True)
    data_batch = Batch.from_data_list(data_object)
    num_atoms = data_batch[0].num_nodes
    return raw_data, data_batch, num_frames, num_atoms


def load_xyz_to_pyg_data(path: Path, atoms_to_graph_kwargs: dict) -> tuple[Atoms, Batch, int, int]:
    """
    Load XYZ data from a given path using ASE and convert it into a list of PyTorch Geometric data objects.

    Args:
        path (Path): Path to the XYZ data file.
        **atoms_to_graph_kwargs: Optional keyword arguments for AtomsToGraphs class.

    Returns:
        Tuple consisting of raw_data, data_batch, num_frames, and num_atoms.
        raw_data (Atoms): Raw data loaded from the file using ASE.
        list_of_data (list): list containing converted data for all frames.
        num_frames (int): Number of frames in the loaded XYZ data file.
        num_atoms (int): Number of atoms in each frame.
    """
    raw_data = ase.io.read(path, index=":")
    num_frames = len(raw_data)
    a2g = AtomsToGraphs(
        **atoms_to_graph_kwargs,
    )
    data_object = a2g.convert_all(raw_data, disable_tqdm=True)
    num_atoms = data_object[0].num_nodes
    return raw_data, data_object, num_frames, num_atoms


def torch_tensor_to_npy(tensor):
    """
    Converts a PyTorch tensor to a numpy array.

    Args:
        tensor (torch.Tensor): PyTorch tensor to be converted.

    Returns:
        Numpy array.
    """
    return tensor.detach().cpu().numpy()
