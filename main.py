import logging
import warnings

# NOTE: Not good practice, disable warnings for now. These are not actionable.
# Could be useful to see warnings in the future.
warnings.filterwarnings("ignore", module="torch_geometric")
warnings.filterwarnings("ignore")

from ocpmodels.common.utils import setup_logging
from ocpmodels.transfer_learning.common.flags import flags
from ocpmodels.transfer_learning.common.utils import get_config
from ocpmodels.transfer_learning.loaders import BaseLoader  # noqa: F401
from ocpmodels.transfer_learning.runners import (
    FTGNNRunner,
    GAPRunner,
    GNNRunner,
    MEKRRGNNRunner,
    MEKRRRunner,
)

if __name__ == "__main__":
    setup_logging()

    parser = flags.get_parser()
    args, override_args = parser.parse_known_args()
    config = get_config(args.config_yml)

    if config["runner"] == "GNN":  # Supervised learning
        runner = GNNRunner(config, args)
    elif config["runner"] == "FTGNN":
        runner = FTGNNRunner(config, args)
    elif config["runner"] == "MEKRR":  # Transfer learning
        runner = MEKRRRunner(config, args)
    elif config["runner"] == "MEKRRGNN":  # Transfer learning
        runner = MEKRRGNNRunner(config, args)
    elif config["runner"] == "GAP":  # Supervised learning
        runner = GAPRunner(config, args)
    else:
        raise NotImplementedError

    runner.setup()
    runner.run()
    logging.info("Done!")
    logging.info(f"Results saved to: {runner.trainer.base_path}")
