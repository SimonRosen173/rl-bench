import json
from typing import Callable, Optional, Dict, Any

from rlbench.utils import join_paths


def proc_env_config_fn_ex(key: str, val: Any, data: Dict) -> Any:
    pass


class Experiment:
    def __init__(
            self,
            base_path: str,
            config_path: str,  # Relative to base_path
            proc_env_config_fn: Optional[Callable] = None,
            preproc_learner_fn: Optional[Callable] = None,
            preproc_env_fn: Optional[Callable] = None,
    ):
        self._proc_env_config_fn = proc_env_config_fn
        self._preproc_learner_fn = preproc_learner_fn
        self._preproc_env_fn = preproc_env_fn

        self._base_path = base_path
        abs_config_path = join_paths(base_path, "exp", config_path)

        self._exp_config = self._process_exp_config(abs_config_path)

    ##################
    # PROCESS CONFIG #
    ##################
    def _process_env_config(
            self,
            env_config_path: str  # Relative to base_path/env
    ):
        abs_env_config_path = join_paths(self._base_path, "env", env_config_path)
        with open(abs_env_config_path, "r") as f:
            env_config = json.load(f)
        return env_config

    def _process_learner_config(
            self,
            learner_config_path: str  # Relative to base_path/learner
    ):
        abs_learner_config_path = join_paths(self._base_path, "learner", learner_config_path)
        with open(abs_learner_config_path, "r") as f:
            learner_config = json.load(f)
        return learner_config

    def _process_exp_config(self, config_path: str) -> Dict:
        abs_config_path = join_paths(self._base_path, "exp", config_path)
        with open(abs_config_path, "r") as f:
            raw_exp_config = json.load(f)

        env_config = self._process_env_config(raw_exp_config["env"]["config"])

        exp_config = {}
        return exp_config
    ##################

    #################
    # SETUP FOLDERS #
    #################
    def setup_folders(self):
        pass

    ##################
    #  CREATE & RUN  #
    ##################
    def create_and_run(self):
        pass

    def rerun_failed(self):
        # Create slurms for failed runs and rerun
        pass

    #######################
    #        RUNS         #
    #######################
    def create_runs(self):
        pass

    def exec_runs(self):
        pass
    #######################

    ############################
    #         SLURMS           #
    # slurm = slurm batch file #
    ############################
    def _create_slurms(self):
        pass

    def exec_slurms(self):
        pass
    ############################


# def proc_env_config():
#     pass
