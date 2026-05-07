import copy
import os
from typing import Generator

import pyrootutils
import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf, open_dict

from cyto_dl.utils.config import kv_to_dict
from scripts.download_test_data import delete_test_data, download_test_data

OmegaConf.register_new_resolver("kv_to_dict", kv_to_dict)
OmegaConf.register_new_resolver("eval", eval)

# Experiment configs to test
experiment_types = [
    "hiera",
    "mae",
    "ijepa",
    "iwm",
    "segmentation",
    "labelfree",
    "gan",
    "instance_seg",
]


@pytest.fixture(scope="package", params=experiment_types)
def cfg_train_global(request) -> DictConfig:
    with initialize(version_base="1.2", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=[
                f"experiment=im2im/{request.param}.yaml",
                "trainer=cpu.yaml",
            ],
        )

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(pyrootutils.find_root())
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_train_batches = 2
            cfg.trainer.limit_val_batches = 2
            cfg.trainer.limit_test_batches = 2
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg


@pytest.fixture(scope="package", params=experiment_types)
def cfg_eval_global(request) -> DictConfig:
    with initialize(version_base="1.2", config_path="../configs"):
        cfg = compose(
            config_name="eval.yaml",
            return_hydra_config=True,
            overrides=[
                "checkpoint.ckpt_path=.",
                f"experiment=im2im/{request.param}.yaml",
                "trainer=cpu.yaml",
            ],
        )

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(pyrootutils.find_root())
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_test_batches = 2
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None
    return cfg


# this is called by each test which uses `cfg_train` arg
# each test generates its own temporary logging path
@pytest.fixture(scope="function")
def cfg_train(cfg_train_global, tmp_path) -> Generator[DictConfig, None, None]:
    cfg = copy.deepcopy(cfg_train_global)

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.data.cache_dir = str(tmp_path / "cache")

    yield cfg

    GlobalHydra.instance().clear()


# this is called by each test which uses `cfg_eval` arg
# each test generates its own temporary logging path
@pytest.fixture(scope="function")
def cfg_eval(cfg_eval_global, tmp_path) -> Generator[DictConfig, None, None]:
    cfg = copy.deepcopy(cfg_eval_global)

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.data.cache_dir = str(tmp_path / "cache")

    yield cfg

    GlobalHydra.instance().clear()


def pytest_sessionstart(session):
    download_test_data()


def pytest_sessionfinish(session, exitstatus):
    # Set CYTO_DL_KEEP_TEST_DATA=1 in CI to preserve the cached download
    # between runs; skipped under pytest-xdist worker shutdown to avoid
    # racing siblings that still hold references.
    if os.environ.get("CYTO_DL_KEEP_TEST_DATA"):
        return exitstatus
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return exitstatus
    delete_test_data()
    return exitstatus


def pytest_collection_modifyitems(config, items):
    # CYTO_DL_SKIP_3D=1 or CYTO_DL_SKIP_2D=1 skips the corresponding
    # spatial_dims parametrizations. PR runs set CYTO_DL_SKIP_3D=1 to keep
    # CI fast; the slow-tests workflow on push-to-main splits the matrix
    # across two parallel jobs (one with each skip flag) to halve wall time.
    skip_3d = os.environ.get("CYTO_DL_SKIP_3D")
    skip_2d = os.environ.get("CYTO_DL_SKIP_2D")
    if not (skip_3d or skip_2d):
        return
    skip_3d_marker = pytest.mark.skip(reason="3D tests skipped (CYTO_DL_SKIP_3D=1)")
    skip_2d_marker = pytest.mark.skip(reason="2D tests skipped (CYTO_DL_SKIP_2D=1)")
    for item in items:
        if skip_3d and (item.name.endswith("-3]") or "-3-" in item.name):
            item.add_marker(skip_3d_marker)
        if skip_2d and (item.name.endswith("-2]") or "-2-" in item.name):
            item.add_marker(skip_2d_marker)
