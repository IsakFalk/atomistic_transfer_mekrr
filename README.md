# Table of Contents

-   [Transfer learning for atomistic simulations using GNNs and kernel mean embeddings](#title)
    -   [How to quickly run](#quickrun)
    -   [Requirements](#requirements)
    -   [Running experiments](#running_exps)
        -   [Configs](#configs)
        -   [Pretrained weights](#pretrained_weights)
        -   [Datasets](#datasets)
    -   [Results](#results)
    -   [Contributing](#contributing)
    -   [Contact](#contact)
    -   [Citing this work](#citation)

<a id="title"></a>

# Transfer learning for atomistic simulations using GNNs and kernel mean embeddings

This repository, forked from the [OCP20
repository](https://github.com/Open-Catalyst-Project/ocp), is the
official implementation and contains the code needed to run the
experiments in the NeurIPS 2023 paper [**Transfer learning for atomistic
simulations using GNNs and kernel mean
embeddings**](https://openreview.net/pdf?id=Enzew8XujO) (ArXiv preprint [**Transfer learning for atomistic
simulations using GNNs and kernel mean
embeddings**](https://arxiv.org/abs/2306.01589)), with the authors

- John (Isak) Falk
- Luigi Bonati
- Pietro Novelli
- Michele Parinello
- Massimiliano Pontil

and carried out at Istituto Italiano di Tecnologia Genoa through a
collaboration between the CSML (John Falk, Pietro Novelli, Massimiliano
Pontil) and Atomistic Simulation groups (Luigi Bonati, Michele
Parinello). The extensions to the original OCP code may be found under
the `ocp/ocpmodels/transfer_learning` and we put our scripts and
experimental scaffolding under the `ocp/transfer_learning` directory.

<a id="quickrun"></a>

## How to quickly run

1.  Set up and activate environment and install necessary libraries

    ``` bash
    micromamba create -f env.yaml -n mekrr
    micromamba activate mekrr

    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
    pip install quippy-ase
    ```

2.  Install package locally

    ``` bash
    pip install .
    ```

3.  Download artefacts (data + weights)

    ``` bash
    make prepare_experiments
    ```

4.  (Optionally) Set up `wandb`

    ``` bash
    wandb login
    ```

    Note that you could also skip directly to 5.

5.  Run methods `GAP, GNN, GNNFT, MEKRR, MEKRR-alpha`

    ``` bash
    python main.py --config-yml transfer_learning/configs/s2ef/cuformate/gap.yaml --cpu
    python main.py --config-yml transfer_learning/configs/s2ef/cuformate/gnn.yaml
    python main.py --config-yml transfer_learning/configs/s2ef/cuformate/gnn-ft.yaml
    python main.py --config-yml transfer_learning/configs/s2ef/cuformate/mekrr.yaml --cpu
    python main.py --config-yml transfer_learning/configs/s2ef/cuformate/mekrr-alpha.yaml --cpu
    ```

<a id="requirements"></a>

## Requirements

You will need a GPU to run these experiments, for running `GAP` and
`MEKRR` we used around 128GB of RAM.

We provide a conda environment file (we recommend using the faster
`micromamba` implementation of the conda API, see [installation
guide](https://mamba.readthedocs.io/en/latest/installation.html), but it
will also work with standard `anaconda`) named `env.yaml` in the root of
the repository. If you are using `conda` replace all instances of
`micromamba` below with `conda`. A conda environment with the necessary
dependencies may be created by running

``` bash
micromamba create -f env.yaml
```

Activate the environment

``` bash
micromamba activate mekrr
```

and install additional packages using pip

``` bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```

Note that we use `wandb` to log results, but it should work without
setting up and account.

<a id="running_exps"></a>

## Running experiments

To download all of the necessary artefacts to run the experiments, run

``` bash
make prepare_experiments
```

which downloads the datasets and the pretrained model weights.

The main experiments are all run through the use of the `main.py`
entrypoint which performs both training and evaluation in the same run.
Run `python main.py --help` for info.

Each experiment is run by dispatching the correct `yaml` file found in
the `transfer_learning/config` directory. The `main.py` function
automatically dispatches the correct code for running each of the
algorithm with pre-specified hyperparameters on the correct dataset
which are completely specified in the config file.

Note that you need to set the correct flags for each of the algorithm
`GAP`, `GNN`, `GNNFT`, `MEKRR` and `MEKRR-alpha` as follows (here using
the provided configs)

``` bash
python main.py --config-yml transfer_learning/configs/s2ef/cuformate/gap.yaml --cpu
python main.py --config-yml transfer_learning/configs/s2ef/cuformate/gnn.yaml
python main.py --config-yml transfer_learning/configs/s2ef/cuformate/gnn-ft.yaml
python main.py --config-yml transfer_learning/configs/s2ef/cuformate/mekrr.yaml --cpu
python main.py --config-yml transfer_learning/configs/s2ef/cuformate/mekrr-alpha.yaml --cpu
```

<a id="configs"></a>

### Configs

The algorithms may be split into 4 different types

- `GAP`
- `GNN`
- `GNNFT`
- `MEKRR` (and `MEKRR-alpha`)

which are indicated by the `runner` key in the config yaml file. Example
config files is found in the directory
`transfer_learning/configs/example` with `base.yaml` consisting of
entries which all algorithms require.

<a id="pretrained_weights"></a>

### Pretrained weights

`MEKRR` and `GNNFT` rely on pretrained weights and the code relies on
the convention of the OCP20 codebase to load these successfully. The
provided makefile has a rule `download_pretrained_weights` which
downloads the weights of the large SchNet model trained on all of the
data (link to
[weights](https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/s2ef/schnet_all_large.pt)
and
[config](https://github.com/Open-Catalyst-Project/ocp/blob/main/configs/s2ef/all/schnet/schnet.yml))
putting it in the correct place for running the experiments defined
through the config files.

<a id="datasets"></a>

### Datasets

After pre-training the feature map, the downstream task consists of
fitting the potential energy surface as a function of the positions and
species. Since we are interested in real-life applications to
heterogonous catalysis, we need to test it on realistic datasets
involving reactive configurations (e.g. in which bonds are broken or
formed). This is a challenging task, and we cannot use standard dataset
such as the ones associated with OCP since they contain configurations
mostly close to equilibrium. For this reason we have selected two
datasets from the literature, the second one which is proprietary but
can either be generated from the same hyperparameters or become
available upon request to any of the (see [Contact](#contact)).

1.  Cu/formate

    Ab initio MD simulations of formate undergoing dehydrogenation
    decomposition (i.e. losing an hydrogen) when interacting with a
    copper surface. We provide a make rule `download_datasets` which
    downloads the cu/formate dataset from the materials cloud repository
    associated with the paper [[1]](#ref1).

2.  Fe/N<sub>2</sub> datasets

    Datasets of increasing complexity related to N2 decomposition on a
    iron surface, including ab initio MD, ab initio metadynamics and
    machine learned based metadynamics for two different system sizes.
    This is more realistic than the previous dataset, and as such we
    heavily rely on it to test the different tasks. You may request the
    data from us, it is the same as that in [[2]](#ref2).

<a id="results"></a>

## Results

These tables mirror those in the preprint

| Algorithm / Dataset | Fe/N<sub>2</sub> D<sub>1</sub> RMSE | Fe/N<sub>2</sub> D<sub>1</sub> MAE | Fe/N<sub>2</sub> D<sub>2</sub> RMSE | Fe/N<sub>2</sub> D<sub>2</sub> MAE | Fe/N<sub>2</sub> D<sub>3</sub> RMSE | Fe/N<sub>2</sub> D<sub>3</sub> MAE | Fe/N<sub>2</sub> D<sub>4</sub> RMSE | Fe/N<sub>2</sub> D<sub>4</sub> MAE | Cu/formate RMSE | Cu/formate MAE |
|---------------------|-------------------------------------|------------------------------------|-------------------------------------|------------------------------------|-------------------------------------|------------------------------------|-------------------------------------|------------------------------------|-----------------|----------------|
| SchNet              | 0.5                                 | 0.4                                | 4.1                                 | 3.2                                | 5.1                                 | 3.8                                | 6.2                                 | 4.7                                | 6.0             | 4.7            |
| GAP-SOAP            | 0.4                                 | 0.4                                | 2.1                                 | 1.5                                | 3.9                                 | 2.9                                | 4.9                                 | 3.0                                | 2.8             | 1.4            |
| SchNet-FT           | **0.1**                             | **0.1**                            | 2.0                                 | 1.5                                | 2.5                                 | 3.2                                | 3.2                                 | 2.6                                | 1.9             | 1.5            |
| MEKRR               | 0.3                                 | 0.3                                | 1.5                                 | 1.2                                | **2.2**                             | **1.7**                            | **2.2**                             | **1.8**                            | 1.7             | 0.9            |
| MEKRR-alpha         | **0.1**                             | **0.1**                            | **1.3**                             | **0.9**                            | 2.4                                 | **1.7**                            | 3.3                                 | 2.3                                | **1.2**         | **0.6**        |

Same-dataset energy prediction. The errors are in units of meV/atom.
Best performance given by **bold**

| Algorithm / Datasets | D<sub>1</sub> → D<sub>2</sub> RMSE | D<sub>1</sub> → D<sub>2</sub> MAE | D<sub>1</sub> → D<sub>3</sub> RMSE | D<sub>1</sub> → D<sub>3</sub> MAE | D<sub>2</sub> → D<sub>3</sub> RMSE | D<sub>2</sub> → D<sub>3</sub> MAE | D<sub>2</sub> → D<sub>4</sub> RMSE | D<sub>2</sub> → D<sub>4</sub> MAE | D<sub>3</sub> → D<sub>4</sub> RMSE | D<sub>3</sub> → D<sub>4</sub> MAE |
|----------------------|------------------------------------|-----------------------------------|------------------------------------|-----------------------------------|------------------------------------|-----------------------------------|------------------------------------|-----------------------------------|------------------------------------|-----------------------------------|
| SchNet               | 13.2                               | 10.1                              | 15.4                               | 12.3                              | 6.2                                | 4.9                               | 93                                 | 90                                | 107                                | 105                               |
| GAP-SOAP             | 24.9                               | 14.6                              | 59.1                               | 34.1                              | 5.8                                | 4.2                               | 830                                | 829                               | 888                                | 888                               |
| SchNet-FT            | 17.6                               | 13.6                              | 27.3                               | 19.4                              | 3.7                                | 2.8                               | 121                                | 119                               | 116                                | 114                               |
| MEKRR                | **8.0**                            | **5.6**                           | **9.3**                            | **6.9**                           | **2.9**                            | **2.2**                           | **27**                             | **20**                            | **55**                             | **51**                            |

Transfer evaluation of algorithms on source to target: D<sub>i</sub> →
D<sub>j</sub>. The errors are in units of meV/atom. Best performance
given by **bold**.

<a id="contributing"></a>

## Contributing

This code is licensed under the MIT license. If you want to contribute
to this repository, please file an issue and / or open up a
pull-request.

<a id="contact"></a>

## Contact

Contact John Falk at `me@isakfalk.com` for general questions or Luigi
Bonati at `luigig.bonati@iit.it` for questions about the
Fe/N<sub>2</sub> datasets.

<a id="citation"></a>

## Citing this work

To cite this work, please use the below bibtex entry

``` example
@inproceedings{
falk2023transfer,
title={Transfer learning for atomistic simulations using {GNN}s and kernel mean embeddings},
author={John Isak Texas Falk and Luigi Bonati and Pietro Novelli and Michele Parrinello and massimiliano pontil},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=Enzew8XujO}
}
```

## References

<a id="ref1">[1]</a> Batzner, S., Musaelian, A., Sun, L., Geiger, M., Mailoa, J. P., Kornbluth, M., ... & Kozinsky, B. (2022). E (3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials. Nature communications, 13(1), 2453, https://www.nature.com/articles/s41467-022-29939-5
<br>
<a id="ref2">[2]</a> Bonati, L., Polino, D., Pizzolitto, C., Biasi, P., Eckert, R., Reitmeier, S., ... & Parrinello, M. (2023). Non-linear temperature dependence of nitrogen adsorption and decomposition on Fe (111) surface, [10.26434/chemrxiv-2023-mlmwv](https://doi.org/10.26434/chemrxiv-2023-mlmwv)
