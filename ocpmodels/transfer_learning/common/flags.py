import argparse
from pathlib import Path


class Flags:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Graph Networks for Electrocatalyst Design")
        self.add_core_args()

    def get_parser(self):
        return self.parser

    def add_core_args(self):
        self.parser.add_argument_group("Core Arguments")
        self.parser.add_argument(
            "--config-yml",
            required=True,
            type=Path,
            help="Path to a config file listing data, model, optim parameters.",
        )
        # self.parser.add_argument(
        #     "--identifier",
        #     default="",
        #     type=str,
        #     help="Experiment identifier to append to checkpoint/log/result directory",
        # )
        self.parser.add_argument(
            "--debug",
            action="store_true",
            help="Whether this is a debugging run or not",
        )
        self.parser.add_argument(
            "--run-dir",
            default="./",
            type=str,
            help="Directory to store checkpoint/log/result directory",
        )
        self.parser.add_argument(
            "--print-every",
            default=10,
            type=int,
            help="Log every N iterations (default: 10)",
        )
        self.parser.add_argument("--seed", default=0, type=int, help="Seed for torch, cuda, numpy")
        # self.parser.add_argument(
        #     "--amp", action="store_true", help="Use mixed-precision training"
        # )
        # self.parser.add_argument(
        #     "--checkpoint", type=str, help="Model checkpoint to load"
        # )
        # self.parser.add_argument(
        #     "--logdir", default="logs", type=Path, help="Where to store logs"
        # )
        self.parser.add_argument("--cpu", action="store_true", help="Run CPU only training")


flags = Flags()
