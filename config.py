import os
from typing import Optional, Union

import yaml


def recursive_merge_dict(base_dict: dict, new_dict: dict):
    """Recursively merge two dictionaries. This performs the union operation (|), but
    is safe for nested dictionaries.

    Args:
        base_dict (dict): base d
        new_dict (dict):

    Returns:
        dict: base dictionary with merged parameters

    """
    for key in new_dict:
        if key in base_dict:
            base_dict[key] = recursive_merge_dict(base_dict[key], new_dict[key])
        else:
            base_dict[key] = new_dict[key]
    return base_dict


def load_config(config: Union[str, dict], config_root: Optional[str] = None):
    """Load a configuration file. If the config has the base_config key
    set, then that base configuration will be loaded recursively. If a
    dictionary is passed, then you should also pass a root directory.
    Otherwise the root directory is inferred from the location of the file.

    Args:
        config (str or dict): Configuration filename, or dictionary
        config_root (str, optional): search path for configurations, defalts to None

    Returns:
        dict: configuration
    """

    if isinstance(config, str):

        if config_root is None:
            config_root = os.path.dirname(config)

        with open(config, "r") as fp:
            config = yaml.load(fp, yaml.SafeLoader)

    elif isinstance(config, dict):
        assert config_root is not None, "Configuration root folder should be specified."
    else:
        raise NotImplementedError(
            "Please provide a dictionary or a path to a config file"
        )

    # Base case, root config
    base_config = config.get("base_config")
    if base_config is not None:

        base_config_path = os.path.join(config_root, base_config)

        # Recurse to load the base configuration
        base_config = load_config(base_config_path)
        config = recursive_merge_dict(base_config, config)

    return config
