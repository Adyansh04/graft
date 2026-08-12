import copy

import pytest
import yaml

CONFIG_PATH = "configs/default.yaml"


@pytest.fixture
def raw_config() -> dict:
    """The shipped default config, as a mutable dict."""
    with open(CONFIG_PATH) as handle:
        return yaml.safe_load(handle)


@pytest.fixture
def config(raw_config):
    from graft.config.schema import Config

    return Config.model_validate(copy.deepcopy(raw_config))
