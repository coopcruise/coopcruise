from datetime import datetime
from functools import partial
import gc
import os
import argparse
import tempfile
import numpy as np
from ray.rllib.algorithms.ppo.ppo import PPO, PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm
from sumo_centralized_envs_new import SumoEnvCentralizedTau, SumoEnvCentralizedVel
from ray.tune.logger import UnifiedLogger
from utils.sim_utils import get_tau_env_config
from ray.tune.registry import register_env
from ray.train.constants import _get_defaults_results_dir


def env_creator_template(config, env_class):
    # This allows access to additional parameters, such as worker_index in the environment config
    return env_class(config)  # Return a gymnasium.Env instance.


# This script implements states, actions and rewards of the MDP model in Section
# 4.4 of the paper, and trains an RL controller to optimize it using the PPO
# algorithm. The hyperparameters are defined in Section 4.4 of the paper, and
# Section 4 of the technical appendix.


NUM_ROLLOUT_WORKERS = 10
INFLOW_TIME_HEADWAY = 2
AV_PERCENT = 100

SINGLE_LANE = False  # True
RIGHT_LANE_CONTROL = False
NUM_CONTROL_SEGMENTS = 2  # 3 # 4 # 5
PER_LANE_CONTROL = False  # True

NUM_SIMULATION_STEPS_PER_STEP = 5
SIMULATION_TIME = 500  # if SINGLE_LANE else 1000

SECONDS_PER_STEP = 0.5


USE_LIBSUMO = True
SHOW_GUI_IN_TRACI_MODE = True

# Set seed to an integer for deterministic simulation. Set to None for
# default behavior.
RANDOM_SEED = None
SUMO_SEED = None  # 0  # None
RL_TRAIN_SEED = None  # 1234  # None
RANDOM_AV_SWITCHING_SEED = None  # 0  # None

CHANGE_LC_AV_ONLY = False
NO_LC = False
NO_LC_RIGHT = True
LC_PARAMS = (
    None  # dict(lcKeepRight=0, lcAssertive=2.5, lcSpeedGain=5, lcImpatience=0.7)
)

ENV_CLS_STR = "SumoEnvCentralizedTau"
ENV_CLS_STR_OPTIONS = ["SumoEnvCentralizedTau", "SumoEnvCentralizedVel"]

DEFAULT_TAU = None
RANDOM_AV_SWITCHING = True
HUMAN_SPEED_STD_0 = True

WARM_UP_TIME = 200  # sec
MERGE_FLOW_DURATION_SINGLE_LANE = 30  # sec
MERGE_FLOW_DURATION_MULTI_LANE = 50  # sec
BREAK_PERIOD_DURATION = 8400  # sec

KEEP_VEH_NAMES_NO_MERGE = True
# Requires creating a different flow file to change that

SCENARIO_DIR = "scenarios/single_junction/rl_scenarios"
# NETWORK_FILE_NAME = None
NETWORK_FILE_NAME = (
    "short_merge_lane_separate_exit_lane_disconnected_merge_lane.net.xml"
)

SCENARIO_PARAMS = {
    # Single junction
    "scenario_start_edge": "992666043",
    "scenario_end_edge": "634155175.210",
    "highway_state_start_edge": "992666043",
    "highway_state_end_edge": "634155175.210",
    # "highway_state_start_edge": "992666042",
    # "highway_state_end_edge": "634155175",
    "control_start_edge": "992666042",
    "control_end_edge": "992666042",
    "state_merge_edges": ["277208926"],
}

ENV_CONFIG_OVERRIDES = {"flat_obs_space": True}
CUSTOM_NAME_POSTFIX = None

DEFAULT_RESULTS_DIR = _get_defaults_results_dir()


def policy_mapping_function(agent_id: str, episode, worker, **kwargs):
    return "hierarchical_policy"


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Train area-based speed traffic congestion controller using PPO.",
        epilog="python3 -i <this-script>",
    )

    # optional input parameters
    parser.add_argument(
        "--env_class",
        type=str,
        default=ENV_CLS_STR,
        help=f"Environment class name. One of: {ENV_CLS_STR_OPTIONS}",
    )

    parser.add_argument(
        "--av_percent",
        type=int,
        default=AV_PERCENT,
        help="Percent of ACC-equipped vehicles.",
    )

    parser.add_argument(
        "--num_control_seg",
        type=int,
        default=NUM_CONTROL_SEGMENTS,
        help="Number of segments before the bottleneck within which to control ACC-equipped vehicles.",
    )

    parser.add_argument(
        "--sim_time",
        type=int,
        default=SIMULATION_TIME,
        help="Simulation time horizon.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=NUM_ROLLOUT_WORKERS,
        help="Number of parallel rollout workers.",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed to use for SUMO, ACC-vehicle choice, and RL training.",
    )

    parser.add_argument(
        "--single_lane",
        default=SINGLE_LANE,
        action="store_true",
        help="Whether to use a single lane scenario",
    )

    parser.add_argument(
        "--right_lane_control",
        default=RIGHT_LANE_CONTROL,
        action="store_true",
        help=(
            "Whether to control only vehicles in the right-most lane. "
            + "Cannot be used together with '--single_lane' or '--per_lane_control'."
        ),
    )

    parser.add_argument(
        "--per_lane_control",
        default=PER_LANE_CONTROL,
        action="store_true",
        help=(
            "Whether to send different control signals to vehicles in different lanes. "
            + "Cannot be used together with '--single_lane' or '--right_lane_control'."
        ),
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help="Path of results directory for storing checkpoints.",
    )

    return parser


if __name__ == "__main__":
    # Specify checkpoint to load algorithm
    checkpoint_path: str | None = None
    # checkpoint_path: str = (
    #     "/home/nz45jg/ray_results/"
    #     "PPO_SumoEnvHierarchicalCentralized_2024-06-23_12-29-09uwabdlev/"
    #     "checkpoint_499"
    # )
    checkpoint_num = 0
    if checkpoint_path is not None:
        algo = Algorithm.from_checkpoint(str(checkpoint_path))
        checkpoint_num = int(str(checkpoint_path).split("checkpoint_")[-1]) + 1

    else:
        parser = create_parser()
        args = parser.parse_args()

        single_lane = args.single_lane
        right_lane_control = args.right_lane_control
        per_lane_control = args.per_lane_control
        av_percent = args.av_percent
        simulation_time = args.sim_time
        random_seed = args.random_seed
        results_dir = args.results_dir
        env_class_str = args.env_class

        if env_class_str == "SumoEnvCentralizedTau":
            env_class = SumoEnvCentralizedTau
        elif env_class_str == "SumoEnvCentralizedVel":
            env_class = SumoEnvCentralizedVel
        else:
            raise ValueError(
                f"env_class argument must be one of: {ENV_CLS_STR_OPTIONS}"
            )

        def logger_creator(config):
            timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
            lane_str = "single_lane" if single_lane else "multi_lane"
            av_percent_str = f"av_{av_percent}"
            seed_str = f"seed_{random_seed}"
            logdir_prefix = (
                f"PPO_{env_class_str}_{lane_str}_{av_percent_str}_{seed_str}_{timestr}"
            )
            if not os.path.exists(results_dir):
                os.makedirs(results_dir, exist_ok=True)
            logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=results_dir)
            return UnifiedLogger(config, logdir, loggers=None)

        random_av_switching_seed = (
            random_seed if random_seed is not None else RANDOM_AV_SWITCHING_SEED
        )
        sumo_seed = random_seed if random_seed is not None else SUMO_SEED
        rl_train_seed = random_seed if random_seed is not None else RL_TRAIN_SEED

        sumo_config_params = {
            "scenario_dir": SCENARIO_DIR,
            "od_flow_file_name": "edge_flows_interval_8400_taz_reduced",
            "single_lane": single_lane,
            "change_lc_av_only": CHANGE_LC_AV_ONLY,
            "no_lc": NO_LC,
            "no_lc_right": NO_LC_RIGHT,
            "lc_params": LC_PARAMS,
            "av_percent": av_percent,
            "warm_up_time": WARM_UP_TIME,
            "merge_flow_duration_single_lane": MERGE_FLOW_DURATION_SINGLE_LANE,
            "merge_flow_duration_multi_lane": MERGE_FLOW_DURATION_MULTI_LANE,
            "break_period_duration": BREAK_PERIOD_DURATION,
            "default_tau": DEFAULT_TAU,
            "keep_veh_names_no_merge": KEEP_VEH_NAMES_NO_MERGE,
            "inflow_time_headway": INFLOW_TIME_HEADWAY,
            "human_speed_std_0": HUMAN_SPEED_STD_0,
            "random_av_switching": RANDOM_AV_SWITCHING,
            "random_av_switching_seed": random_av_switching_seed,
        }

        if NETWORK_FILE_NAME is not None:
            sumo_config_params.update({"network_file_name": NETWORK_FILE_NAME})

        env_config_overrides = ENV_CONFIG_OVERRIDES | {
            "right_lane_control": right_lane_control
        }

        sim_config_params = {
            "use_libsumo": USE_LIBSUMO,
            "show_gui_in_traci_mode": SHOW_GUI_IN_TRACI_MODE,
            "sumo_seed": sumo_seed,
            "seconds_per_step": SECONDS_PER_STEP,
            "num_simulation_steps_per_step": NUM_SIMULATION_STEPS_PER_STEP,
            "simulation_time": simulation_time,
            "scenario_params": SCENARIO_PARAMS,
            "env_config_overrides": env_config_overrides,
            "custom_name_postfix": CUSTOM_NAME_POSTFIX,
            "num_control_segments": NUM_CONTROL_SEGMENTS,
            "per_lane_control": per_lane_control,
            "with_eval": False,
        }

        num_simulation_steps = int(SIMULATION_TIME / SECONDS_PER_STEP)

        env_config = get_tau_env_config(sumo_config_params, sim_config_params)
        # If False, uses environment steps to count rollout and batch steps. Each
        # environment step can include many agent (autonomous vehicle) steps.
        count_steps_by_agent = False

        # If 0, uses the default worker for rollouts. If larger than 0, creates
        # separate worker instances for each rollout worker.
        num_rollout_workers = args.num_workers
        def_batch_size = 4000
        rollout_fragment_length = (
            int(def_batch_size / max(1, num_rollout_workers))
            if count_steps_by_agent
            else num_simulation_steps
        )

        rollout_fragment_length = int(
            rollout_fragment_length / NUM_SIMULATION_STEPS_PER_STEP
        )
        # A training step is run on a batch of experience.
        #
        # For PPO specifically: In each training step the weights are updated
        # multiple time on mini batches of 128 agent steps by default.
        #
        # Training seems to work even if the episode is not done. The episode is
        # done only when `truncated["__all__"]` or `terminated["__all__"]` are
        # `True`.
        #
        # However, evaluation metrics are computed only for episodes that are done.
        # See the methods step, `_process_observations`, `_handle_done_episode`, and
        # `_get_rollout_metrics` in ray.rllib.evaluation.env_runner_v2.py. Only when
        # the episode is done, the environment runner `step` function will return a
        # `RolloutMetrics` object. Only then it is put into the metrics queue of the
        # sampler object calling the env runner (see the get_data method in
        # ray.rllib.evaluation.sampler.py).
        #
        # Sampling is done by each rollout worker in its `sample` method (see
        # `RolloutWorker` class in ray.rllib.evaluation.rollout_worker.py). It then
        # gets a batch from a sampler (`SamplerInput`) using its `next` function,
        # which calls the `get_data` method.
        train_batch_size = rollout_fragment_length * max(1, num_rollout_workers)

        # The following environment implements MDP states, actions, and rewards
        # (see Section 4.4 in the paper), and interfaces the SUMO simulator. The
        # environment initializes the simulation when reset() is called, and
        # advances the simulation when step() is called, using the specified
        # actions.
        # Register environment
        env_creator = partial(env_creator_template, env_class=env_class)
        register_env(env_class_str, env_creator)
        # Create an instance of the environment for extracting observation and action spaces
        env_instance = env_creator(env_config)

        alg_config = (
            PPOConfig()
            .environment(env=env_class_str, env_config=env_config)
            .rollouts(
                num_rollout_workers=num_rollout_workers,
                rollout_fragment_length=rollout_fragment_length,
                create_env_on_local_worker=True,
            )
            # PPO-specific training parameters
            .training(
                # enable_learner_api=True,
                train_batch_size=train_batch_size,
                # lambda_=0.97,
                # gamma=0.99,
                # clip_param=0.2,
                # num_sgd_iter=10,
                # use_gae=True,
                # kl_target=0.2,
                # entropy_coeff=0.001,
            )
            # If using multi-agent, configure multi-agent settings:
            .multi_agent(
                policy_mapping_fn=policy_mapping_function,
                policies={
                    "hierarchical_policy": (
                        None,
                        env_instance.observation_space,
                        env_instance.action_space,
                        # env_cls(env_config).observation_space,
                        # env_cls(env_config).action_space,
                        {},
                    ),
                },
            )
            .debugging(
                seed=RL_TRAIN_SEED,
            )
        )
        alg_config._disable_preprocessor_api = True

        algo = PPO(alg_config, logger_creator=logger_creator)

    num_training_steps = 10000  # 2500
    num_steps_between_saves = 250  # 50
    best_reward = -np.inf  # TODO: Change for loading case
    checkpoint_dir = "checkpoints"
    for train_step in range(num_training_steps):
        training_progress = algo.train()
        print(f"Completed training step #{train_step}")
        # print(training_progress)

        # Call `save()` to create a checkpoint.
        reward_mean = training_progress["sampler_results"]["episode_reward_mean"]
        if reward_mean > best_reward:
            best_reward = reward_mean
            algo.save(checkpoint_dir=os.path.join(algo.logdir, "checkpoint_best"))

        if (
            checkpoint_num + train_step
        ) % num_steps_between_saves == 0 or train_step == num_training_steps - 1:
            save_result = algo.save(
                checkpoint_dir=os.path.join(
                    algo.logdir, f"checkpoint_{checkpoint_num + train_step}"
                )
            )
            path_to_checkpoint = save_result.checkpoint.path
            print(f"{save_result = }")
            print(
                "An Algorithm checkpoint has been created inside directory: "
                f"'{path_to_checkpoint}'."
            )
        gc.collect()
    # If evaluation_duration_unit is "episodes", the algorithm runs
    # evaluation_duration episodes with rollout_fragment_length steps. If
    # count_steps_by is "env_steps", counts each environment step as a single
    # step, and it does not matter how many agents are in the environment. If
    # count_steps_by is "agent_steps", each agent step is counted as a single
    # step.

    # eval = algo.evaluate()
    # print(eval)

    # Terminate the algo
    algo.stop()
