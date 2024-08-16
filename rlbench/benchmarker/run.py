from typing import Callable, Dict, Any, Union, Optional


class RunConfig:
    def __init__(
            self,
            config_path: str
    ):
        import pickle
        with open(config_path, "rb") as f:
            config = pickle.load(f)

        self.run_path = config_path
        self.exp_config_path = config["exp"]["config_path"]

        self.seed = config["run"]["seed"]
        self.exp_type = config["run"]["exp_type"]
        self.repeat_no = config["run"]["repeat_no"]
        self.run_no = config["run"]["run_no"]

        self.learner_name = config["learner"]["name"]
        self.env_name = config["env"]["name"]

        # kwargs
        self.learner_kwargs = config["learner"]["kwargs"]
        self.env_kwargs = config["learner"]["kwargs"]
        self.logger_kwargs = config["logger"]["kwargs"]

        # ...

    def to_json_dict(self):
        # TODO
        raise NotImplementedError


class Run:
    def __init__(
            self,
            runnable_fn: Callable,
            run_config: Union[str, RunConfig],
            node: Optional[str] = None
    ):
        """
        :param runnable_fn: function that takes RunConfig object & Logger object as only arguments
        :param run_config:
        :param node:
        """
        self.node = node
        self.hardware_specs = self._get_hardware_specs()

        if type(run_config) == str:
            run_config = RunConfig(run_config)
        self.run_config: RunConfig = run_config

        self._setup()

        self.logger = ...

        self._is_entered = False
        self._is_finished = False

    def _setup(self):
        pass

    def _get_hardware_specs(self):
        raise NotImplementedError

    def run(self):
        pass

    def finish(self):
        pass
        # TODO: Indicate that run succeeded

        if self._is_finished:
            raise RuntimeError("Run.finish() can only be called once per Run object")

        self._is_finished = True

    #################
    # MAGIC METHODS #
    #################
    def __enter__(self):
        self.is_entered = True
        # Do something?

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()
    #################
