"""Config file parsing tools"""
import os
from typing import Optional, Union

import pkg_resources
import yaml


def recursive_merge_dict(base_dict: dict, new_dict: dict):
    """Recursively merge two dictionaries. This performs the union operation (|), but
    is safe for nested dictionaries.

    Args:
        base_dict (dict): base dictionary
        new_dict (dict): dictionary to merge into base

    Returns:
        dict: base dictionary with merged parameters

    """

    assert isinstance(base_dict, dict)
    assert isinstance(new_dict, dict)

    for key in new_dict:
        if key in base_dict:
            if isinstance(base_dict[key], dict):
                base_dict[key] = recursive_merge_dict(base_dict[key], new_dict[key])
            else:
                base_dict[key] = new_dict[key]
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

    Raises:
        NotImplementedError: If a dict or a string isn't provided
    """

    if isinstance(config, str):
        if config_root is None:
            config_root = os.path.dirname(config)

        with open(config, "r", encoding="utf-8") as fp:
            config = yaml.load(fp, yaml.SafeLoader)

    elif isinstance(config, dict):
        config_root = config["config_root"]
    else:
        raise NotImplementedError(
            "Please provide a dictionary or a path to a config file"
        )

    assert isinstance(config, dict)

    # Need to merge configurations
    base_config_path = config.get("base_config")
    if base_config_path is not None:
        base_config_path = os.path.join(config_root, base_config_path)

        # Recurse to load the base configuration
        base_config = load_config(base_config_path, config_root)

        assert isinstance(base_config, dict)
        assert isinstance(config, dict)

        return recursive_merge_dict(base_config, config)

    # Base case, root configuration.

    config["config_root"] = config_root

    return config
