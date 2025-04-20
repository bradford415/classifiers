# Utility functions to reproduce the results from experimentss
import json
import logging
import random
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import yaml

log = logging.getLogger(__name__)


def reproducibility(seed: int) -> None:
    """Set the seed for the sources of randomization. This allows for more reproducible results"""

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def model_info(model):
    model_size = 0
    for param in model.parameters():
        model_size += param.data.nelement()
    log.info("Model params: %.2f M", (model_size / 1024 / 1024))


def save_configs(
    config_dicts: Sequence[tuple[Dict, str]],
    solver_dict: tuple[dict, str],
    output_path: Path,
):
    """Save configuration dictionaries as json files in the output; this allows
    reproducibility of the model by saving the parameters used

    Args:
        config_dicts: a sequence of tuples where each tuple contains a dict of the configuration parameters
                      used to to run the script (e.g., the base config and the model config) and a str for the
                      name to save the json file as
        output_path: Output directory to save the configuration files; it's recommened to have the
                     final dir named "reproduce"
    """

    output_path.mkdir(parents=True, exist_ok=True)

    # Save yaml configurations
    for config_dict, save_name in config_dicts:
        with open(output_path / save_name, "w") as f:
            yaml.dump(
                config_dict, f, indent=4, sort_keys=False, default_flow_style=False
            )

    # Save solver parameters (optimizer, lr_scheduler, etc.)
    param_dict, save_name = solver_dict
    with open(output_path / save_name, "w") as f:
        json.dump(param_dict, f, indent=4)
