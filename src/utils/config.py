import os
import yaml

def load_and_merge_configs(config_folder: str) -> dict:
    merged_config = {}

    # List all yaml files in the folder
    yaml_files = sorted(
        f for f in os.listdir(config_folder) if f.endswith((".yaml", ".yml"))
    )

    for yaml_file in yaml_files:
        path = os.path.join(config_folder, yaml_file)
        with open(path, "r") as f:
            config_part = yaml.safe_load(f)
            # Merge new config dictionary into the merged config
            merged_config = deep_merge_dicts(merged_config, config_part)

    return merged_config

def deep_merge_dicts(dict1, dict2):
    """
    Recursively merge dict2 into dict1 and return the merged dict.
    Values in dict2 override those in dict1.
    """
    for key, val in dict2.items():
        if (
            key in dict1
            and isinstance(dict1[key], dict)
            and isinstance(val, dict)
        ):
            dict1[key] = deep_merge_dicts(dict1[key], val)
        else:
            dict1[key] = val
    return dict1
