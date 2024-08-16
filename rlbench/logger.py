import copy
import json
import os
import shutil
from collections import defaultdict
from enum import IntEnum
from typing import List, Optional, Dict, Tuple, Any, Union, SupportsInt, SupportsFloat

import typing

import pandas as pd
import wandb

from rlbench.utils import join_paths, create_or_clear_folder, save_json, copy_to_dict


class Logger:
    def __init__(
            self,
            log_mode: str,
            # metrics: Dict[str, List[Tuple[str, str]]],
            metrics: Dict[str, Tuple[str, List[str]]],
            name: str,
            job_type: Optional[str] = None,
            project: Optional[str] = None,  # Required for WANDB
            reinit: bool = True,
            group: Optional[str] = None,
            tags: Optional[List[str]] = None,
            entity: Optional[str] = None,
            notes: Optional[str] = None,
            config: Optional[Dict[str, Any]] = None,
            # BINNING
            bin_size: Optional[Union[int, Dict[str, int]]] = None,
            # WANDB
            wandb_mode: str = "online",
            # LOCAL
            local_base_dir: Optional[str] = None,
            # SUB LOGGERS
            is_sub_logger: bool = False,
            sub_loggers: Optional[List] = None
    ):
        """
        :param log_mode:
        :param metrics: Must be dict of form {'group_1': ('step_metric', ['metric_1', 'metric_2', ...]), 'group_2': ('step_metric', ['metric_1', 'metric_2', ...])}
        :param name:
        :param job_type:
        :param project:
        :param reinit:
        :param group:
        :param tags:
        :param entity:
        :param notes:
        :param config:
        :param bin_size:
        :param wandb_mode:
        :param local_dir:
        :param is_sub_logger:
        :param sub_loggers:
        """
        ############
        # LOG MODE #
        ############
        log_mode_valid_vals = ["disabled", "wandb", "local"]
        if log_mode == "disabled":
            self._log_mode = LogMode.DISABLED
        elif log_mode == "wandb":
            self._log_mode = LogMode.WANDB
        elif log_mode == "local":
            self._log_mode = LogMode.LOCAL
        else:
            raise ValueError(f"Invalid value for log_mode={log_mode}. Valid values = {log_mode_valid_vals}")

        self._is_enabled = self._log_mode != LogMode.DISABLED

        ############
        # GEN VARS #
        ############
        self._name = name

        self._job_type = job_type
        self._project = project
        self._group = group
        self._entity = entity
        self._tags = tags
        self._notes = notes

        self._config = conv_to_json_d(config)

        self._reinit = reinit

        ##############
        # SUB LOGGER #
        ##############
        self.is_sub_logger = is_sub_logger
        self.has_sub_loggers = sub_loggers is not None
        if self.is_sub_logger and self.has_sub_loggers:
            raise ValueError("Logger cannot have sub_loggers if is_sub_logger=True")

        if not self.has_sub_loggers:
            self._sub_loggers = []
        else:
            self._sub_loggers = sub_loggers
        ##############

        ###########
        # METRICS #
        ###########
        self._metric_groups = list(metrics.keys())
        self._step_metrics = {metric_group: tup[0] for metric_group, tup in metrics.items()}
        self._metrics = {metric_group: tup[1] for metric_group, tup in metrics.items()}
        # Remove repeats and remove step metric from metrics if there
        for metric_group, curr_metrics in self._metrics.items():
            curr_metrics = set(curr_metrics).difference({self._step_metrics[metric_group]})
            self._metrics[metric_group] = list(curr_metrics)

        ###########
        # SUMMARY #
        ###########
        self._summary = None
        if self._is_enabled and self._log_mode == LogMode.LOCAL:
            self._summary = {metric_group: {metric: None for metric in metrics_arr}
                             for metric_group, metrics_arr in self._metrics.items()}
            for metric_group, step_metric in self._step_metrics.items():
                self._summary[metric_group][step_metric] = None

        ##################
        # WANDB SPECIFIC #
        ##################
        valid_wandb_modes = ["online", "local", "disabled"]
        if wandb_mode not in valid_wandb_modes:
            raise ValueError(f"Invalid value for wandb_mode={wandb_mode}. Valid values = {valid_wandb_modes}")
        self._wandb_mode = wandb_mode
        self._wandb_run: Optional[wandb.run] = None

        ###########
        # BINNING #
        ###########
        if bin_size is None:
            self._bin_size = None
        elif type(bin_size) == int:
            self._bin_size = {group: bin_size for group in self._metric_groups}
        elif type(bin_size) == dict:
            if set(bin_size.keys()) != set(self._metric_groups):
                raise ValueError("Keys of bin_size must be same as keys of metrics")
            self._bin_size = bin_size
        else:
            raise TypeError(f"Wrong type for bin_size. type(bin_size)={type(bin_size)}")

        self._metric_roll_avg = None
        self._metric_n = None
        # if self._bin_size is not None:
        #     self._metric_roll_avg = {metric_group: {metric: 0 for metric in self._metrics[group]} for
        #                              metric_group in self._metric_groups}
        #     self._metric_n = {metric_group: {metric: 0 for metric in self._metrics[group]} for
        #                       metric_group in self._metric_groups}

        #######################
        # DIRECTORIES & PATHS #
        #######################
        # Directories
        self._local_base_dir = local_base_dir
        self._run_dir = None
        self._files_dir = None
        self._artifacts_dir = None
        self._metrics_dir = None
        self._temp_dir = None
        # Other paths
        self._metrics_file_paths: Optional[Dict[str, str]] = None
        self._summary_path: Optional[str] = None

        #########
        # FILES #
        #########
        self._metrics_files: Optional[Dict[str, typing.TextIO]] = None

        #########
        # OTHER #
        #########
        self._is_finished = False

        #########
        # SETUP #
        #########
        if self._is_enabled:
            self._setup()

    #########
    # SETUP #
    #########
    def _setup(self, reinit: Optional[bool] = None):
        if self._is_enabled:
            # BINNING
            if self._bin_size is not None:
                self._metric_roll_avg = {metric_group: {metric: 0 for metric in self._metrics[metric_group]} for
                                         metric_group in self._metric_groups}
                self._metric_n = {metric_group: {metric: 0 for metric in self._metrics[metric_group]} for
                                  metric_group in self._metric_groups}

            if reinit is None:
                reinit = self._reinit

            # SETUP LOCAL OR WANDB
            if self._log_mode == LogMode.LOCAL:
                self._setup_local(reinit)
            elif self._log_mode == LogMode.WANDB:
                self._setup_wandb(reinit)
        # TODO: Handle reinit

    # def _setup_metrics(self):
    #     if self._is_enabled:
    #         pass

    # LOCAL #
    def _setup_local(
            self,
            reinit=True
    ):
        if not reinit:
            raise NotImplementedError("reinit=False not currently implemented")

        ###############
        # SETUP PATHS #
        ###############
        self._run_dir = join_paths(self._local_base_dir, self._name)
        self._summary_path = join_paths(self._run_dir, "summary.json")

        self._temp_dir = join_paths(self._run_dir, "temp")
        self._files_dir = join_paths(self._run_dir, "files")
        self._artifacts_dir = join_paths(self._run_dir, "artifacts")
        self._metrics_dir = join_paths(self._run_dir, "metrics")

        if os.path.isdir(self._run_dir):
            print(f"WARNING: run_dir={self._run_dir} existed and was cleared")
        create_or_clear_folder(self._run_dir)

        os.mkdir(self._temp_dir)
        os.mkdir(self._files_dir)
        os.mkdir(self._artifacts_dir)
        os.mkdir(self._metrics_dir)

        # create_or_clear_folder(self._temp_dir)
        # create_or_clear_folder(self._files_dir)
        # create_or_clear_folder(self._artifacts_dir)
        # create_or_clear_folder(self._metrics_dir)
        ###############

        #######################
        # CONFIG & 'METADATA' #
        #######################
        config_path = join_paths(self._run_dir, "config.json")
        # with open(config_path, "w") as f:
        #     json.dump(self._config, f, indent=1)
        save_json(self._config, config_path)

        metadata_d = {
            "name": self._name,
            "job_type": self._job_type,
            "project": self._project,
            "group": self._group,
            "entity": self._entity,
            "tags": self._tags,
            "notes": self._notes
        }
        metadata_path = join_paths(self._run_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata_d, f, indent=1)
        #######################

        ###########
        # METRICS #
        ###########
        self._metrics_file_paths = {}
        self._metrics_files = {}
        for metric_group in self._metric_groups:
            curr_path = join_paths(self._metrics_dir, f"{metric_group}.csv")
            curr_file = open(curr_path, "w")
            self._metrics_file_paths[metric_group] = curr_path
            self._metrics_files[metric_group] = curr_file

            if len(self._metrics[metric_group]) == 0:
                raise ValueError(f"self._metrics[{metric_group}] is empty")

            cols_str = f"{self._step_metrics[metric_group]}," + ",".join(self._metrics[metric_group])
            curr_file.write(cols_str)
        ###########

        ##########
        # STATUS #
        ##########
        status_path = join_paths(self._run_dir, "status.txt")
        with open(status_path, "w") as f:
            f.write("started")

    # WANDB #
    def _setup_wandb(
            self,
            reinit=True
    ):
        self._wandb_run: wandb.run = wandb.init(
            job_type=self._job_type,
            config=self._config,
            project=self._project,
            entity=self._entity,
            reinit=reinit,
            tags=self._tags,
            group=self._group,
            notes=self._notes,
            mode=self._wandb_mode
        )

        ###############
        # SETUP PATHS #
        ###############
        self._run_dir = self._wandb_run.dir
        self._temp_dir = join_paths(self._run_dir, "temp")
        self._files_dir = join_paths(self._run_dir, "files")
        self._artifacts_dir = join_paths(self._run_dir, "artifacts")

        create_or_clear_folder(self._temp_dir)
        create_or_clear_folder(self._files_dir)
        create_or_clear_folder(self._artifacts_dir)
        ###############

        #################
        # SETUP METRICS #
        #################
        for metric_group, metrics_arr in self._metrics.items():
            for name, step_metric in metrics_arr:
                self._wandb_run.define_metric(name, step_metric=step_metric)
        #################
        pass

    #########

    def add_metrics(
            self,
            metrics: Dict[str, List[Tuple[str, str]]]
    ):
        # TODO
        raise NotImplementedError

        if self._log_mode == LogMode.WANDB:
            for metrics_group, metrics_arr in metrics.items():
                for name, step_metric in metrics_arr:
                    self._wandb_run.define_metric(name, step_metric=step_metric)

                if metrics_group in self._metrics:
                    self._metrics[metrics_group].extend(metrics_arr)
                else:
                    self._metrics[metrics_group] = metrics_arr

        elif self._log_mode == LogMode.LOCAL:
            for metrics_group, metrics_arr in metrics.items():
                if metrics_group in self._metrics:
                    # self._metrics[metrics_group].extend(metrics_arr)
                    #
                    # # add to existing metrics file
                    # curr_metrics_file = self._metrics_files[metrics_group]
                    # curr_metrics_file_path = self._metrics_file_paths[metrics_group]
                    #
                    # curr_metrics_file.close()
                    # df = pd.read_csv(curr_metrics_file_path)
                    # metric_cols = [tup[0] for tup in metrics_arr]
                    # # self.metric_vals.extend()
                    # df[metric_cols] = None
                    # self._metrics[metrics_group] = list(df.columns)
                    #
                    # df.to_csv(curr_metrics_file_path, index=False)
                    # self._metrics_files[metrics_group] = open(curr_metrics_file_path, "a")

                    raise NotImplementedError("This has not been tested yet")
                else:
                    self._step_metrics[metrics_group] = metrics_arr[0][1]

                    self.metrics[metrics_group] = metrics_arr
                    self.metric_vals[metrics_group] = [tup[0] for tup in metrics_arr]

                    # create new metrics file
                    curr_path = join_paths([self.metrics_dir, f"{metrics_group}.csv"])
                    curr_file = open(curr_path, "w")
                    self.metrics_file_paths[metrics_group] = curr_path
                    self.metrics_files[metrics_group] = curr_file

                    cols_str = ",".join(self.metric_vals[metrics_group])
                    curr_file.write(cols_str)

                    # print("TODO: Test Logger.add_metrics()")
        elif self._log_mode == LogMode.DISABLED:
            return

        self._metric_groups = list(self._metrics.keys())

    ###########
    # LOGGING #
    ###########
    def log(
            self,
            d: Dict[str, Union[SupportsInt, SupportsFloat]],
            group: str,
            check_keys: bool = False
    ):
        if self._is_enabled:
            if group not in self._metrics:
                raise ValueError(f"Invalid value for group={group} given. "
                                 f"Valid values = {list(self._metrics.keys())}")

            if check_keys:
                # TODO: Check
                # s1 = set(d.keys())
                # step_metric_s = {self._step_metrics[group]}
                s = set(d.keys()).difference({self._step_metrics[group]})
                set_diff = s.difference(set(self._metrics[group]))
                if len(set_diff) != 0:
                    raise ValueError(f"Invalid keys specified in d. Invalid keys = {set_diff}")

            if self._log_mode == LogMode.LOCAL:
                self._log_local(d, group, check_keys=False)
            elif self._log_mode == LogMode.WANDB:
                self._log_wandb(d)
            else:
                # This should never be entered, but error thrown just in case
                raise ValueError

    def log_binned(
            self,
            d: Dict[str, Union[SupportsInt, SupportsFloat]],
            group: str
    ):
        if self._is_enabled:
            if self._bin_size is None:
                raise ValueError(f"self._bin_size cannot be equal to None when calling log_binned")
            if group not in self._metrics:
                raise ValueError(f"Invalid value for group = {group}")

            if set(d.keys()) != set(self._metrics[group]):
                raise ValueError("All metrics in group must be given when binning")

            group_bin_size = self._bin_size[group]

            # TODO: Check this is correct
            n = self._metric_n[group] + 1
            for metric, val in d.items():
                curr_avg = self._metric_roll_avg[group][metric]
                # Calc rolling average iteratively
                new_avg = curr_avg * (n-1)/n + val/n

                self._metric_roll_avg[group][metric] = new_avg

            if n == group_bin_size:
                avg_d = copy.deepcopy(self._metric_roll_avg[group])
                self._metric_n[group] = 0
                self.log(avg_d, group)

    def log_to_file_pkl(
            self,
            data: Any,
            file_name: str
    ):
        raise NotImplementedError

    def log_file(
            self,
            file_path: str
    ):
        if self._is_enabled:
            if self._log_mode == LogMode.LOCAL:
                self._log_file_local(file_path)
            elif self._log_mode == LogMode.WANDB:
                self._log_file_wandb(file_path)

    def log_artifact(
            self,
            file_path: str,
            name: Optional[str] = None,
            artifact_type: Optional[str] = None,
            aliases: Optional[List[str]] = None
    ):
        raise NotImplementedError

    def log_summary(self):
        if self._is_enabled and self._log_mode == LogMode.LOCAL:
            if self._summary is None:
                raise ValueError("summary has not been set")
            save_json(self._summary, self._summary_path)

    # ----- #
    # LOCAL #
    # ----- #
    def _to_row(
            self,
            d: Dict[str, Union[SupportsInt, SupportsFloat]],
            group: str,
            check_keys: bool = False
    ):
        group_metrics = self._metrics[group]
        step_metric = self._step_metrics[group]

        if step_metric not in d:
            raise ValueError(f"step_metric={step_metric} not specified in d")

        # Check if non-registered keys are in d
        if check_keys:
            set_diff = set(d.keys()) - set(group_metrics)
            if len(set_diff) == 0:
                raise ValueError(f"Unregistered keys present in d. Unregistered keys = {set_diff}")

        dd = defaultdict(lambda: "", d)
        arr = [str(dd[col]) for col in self._metrics[group]]
        row = f"{str(d[step_metric])}," + ",".join(arr)
        return row

    def _log_local(
            self,
            d: Dict[str, Union[SupportsInt, SupportsFloat]],
            group: str,
            check_keys: bool = False
    ):
        if group not in self._metrics:
            raise ValueError(f"Invalid value for group={group}")
        copy_to_dict(from_dict=d, to_dict=self._summary[group])

        row = self._to_row(d, group, check_keys)
        self._metrics_files[group].write(f"\n{row}")

    def _log_file_local(
            self,
            file_path: str
    ):
        file_name = os.path.basename(file_path)
        new_path = join_paths(self._files_dir, file_name)
        if os.path.exists(new_path):
            print(f"WARNING: {file_name} already in files dir so skipped.")
        else:
            shutil.copy(src=file_path, dst=self._files_dir)

    def _log_artifact_local(self):
        raise NotImplementedError

    # ----- #

    # ----- #
    # WANDB #
    # ----- #
    def _log_wandb(
            self,
            d: Dict[str, Union[SupportsInt, SupportsFloat]],
            commit: bool = True
    ):
        self._wandb_run.log(data=d, commit=commit)

    def _log_file_wandb(
            self,
            file_path: str
    ):
        # TODO: Test
        file_name = os.path.basename(file_path)
        new_path = join_paths(self._files_dir, file_name)
        # wandb_path = join_paths([self._wandb_dir, file_name])
        shutil.copy(file_path, new_path)
        self._wandb_run.save(new_path, base_path=self._files_dir, policy="live")

    def _log_artifact_wandb(self):
        raise NotImplementedError
    ###########

    ##########
    # FINISH #
    ##########

    def finish(self):
        if self._is_enabled:
            if self._is_finished:
                raise RuntimeError("Run is already finished. finish cannot be called again.")
            if self._log_mode == LogMode.LOCAL:
                # Close metric files (assume open)
                for curr_file in self._metrics_files.values():
                    curr_file.close()

                # SUMMARY #
                self.log_summary()

                ##########
                # STATUS #
                ##########
                status_path = join_paths(self._run_dir, "status.txt")
                with open(status_path, "w") as f:
                    f.write("finished")
            elif self._log_mode == LogMode.WANDB:
                self._wandb_run.finish()


class LogMode(IntEnum):
    DISABLED = 0
    WANDB = 1
    LOCAL = 2


def conv_to_json_d(x):
    # raise NotImplementedError
    print("WARNING: conv_to_json_d not properly implemented")
    return x


def test():
    metrics = {
        "train": ('step', ['metric1', 'metric2'])
    }
    logger = Logger(
        log_mode="local",
        metrics=metrics, name="debug",
        local_base_dir="debug_data"
    )

    log_d = {
        "step": 1,
        "metric1": 10
    }
    logger.log(log_d, group="train", check_keys=True)
    # logger.log_summary()

    logger.finish()
    pass


def main():
    test()


if __name__ == "__main__":
    main()
