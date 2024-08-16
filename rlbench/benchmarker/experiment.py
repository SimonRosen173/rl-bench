from typing import Callable


class Experiment:
    def __init__(
            self,
            config_path: str,
            proc_env_config_fn: Callable,
            preproc_learner_fn: Callable,
            preproc_env_fn: Callable,
    ):
        pass

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
