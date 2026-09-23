import gc
import json
import argparse
import os

# import time
import numpy as np

from pathlib import Path

import multiprocessing

# from multiprocessing import Process, Queue
import queue  # imported for using queue.Empty exception
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.spaces.space_utils import unsquash_action
from ray.rllib.env.env_context import EnvContext
import traci.constants as tc

from sumo_centralized_envs_new import (
    SumoEnvCentralizedMinGap,
    SumoEnvCentralizedTau,
    SumoEnvCentralizedVel,
)

from train_ppo_centralized import (
    ENV_CLS_STR_OPTIONS,
    get_env_class_from_str,
    add_parser_simulation_params,
    DEF_SUMO_CONFIG_PARAMS,
    DEF_SIM_CONFIG_PARAMS,
)

# from train_ppo_centralized import create_parser
from utils.analysis_utils import analyze_sim_group
from utils.sumo_utils import extract_vehicle_ids_from_routes, get_episode_results_dir
from utils.sim_utils import DEF_SUMO_CONFIG, get_centralized_env_config

NUM_ROLLOUT_WORKERS = 10
# INFLOW_TIME_HEADWAY = 2
# AV_PERCENT = 100

# SINGLE_LANE = False  # True
# NUM_CONTROL_SEGMENTS = 2  # 3 # 4 # 5
PER_LANE_CONTROL = False  # True

# NUM_SIMULATION_STEPS_PER_STEP = 5
# SIMULATION_TIME = 500  # if SINGLE_LANE else 1000

# SECONDS_PER_STEP = 0.5


# USE_LIBSUMO = True
# SHOW_GUI_IN_TRACI_MODE = True

# Set seed to an integer for deterministic simulation. Set to None for
# default behavior.
# RANDOM_SEED = None
SUMO_SEED = None  # 0  # None
# RANDOM_AV_SWITCHING_SEED = None  # 0  # None

NUM_TESTS = 30

RESULTS_DIR = "test"

# CHANGE_LC_AV_ONLY = False
# NO_LC = False
# NO_LC_RIGHT = True
# LC_PARAMS = (
#     None  # dict(lcKeepRight=0, lcAssertive=2.5, lcSpeedGain=5, lcImpatience=0.7)
# )
# DEFAULT_TAU = None
# RANDOM_AV_SWITCHING = True
# HUMAN_SPEED_STD_0 = True

# WARM_UP_TIME = 200  # sec
# MERGE_FLOW_DURATION_SINGLE_LANE = 30  # sec
# MERGE_FLOW_DURATION_MULTI_LANE = 50  # sec
# BREAK_PERIOD_DURATION = 8400  # sec

# KEEP_VEH_NAMES_NO_MERGE = True

# NETWORK_FILE_NAME = None
# NETWORK_FILE_NAME = (
#     "short_merge_lane_separate_exit_lane_disconnected_merge_lane.net.xml"
# )

DEF_RL_CONTROL_PARAMS = {
    "alg_checkpoint_path": None,  # Must be overridden to use RL control
    "rl_per_lane_control": False,
}

# DEF_TAU_CONTROL_CONSTANTS = {
#     "tau_control_only_rightmost_lane": True,
#     # The following is not relevant if automatic_tau_duration=True
#     "tau_control_start_time": 200,
#     "tau_control_duration": 100,
# }


def create_eval_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Evaluate traffic congestion controller",
        epilog="python3 -i <this-script>",
    )

    parser.add_argument(
        "path", type=str, nargs="?", default=None, help="Path to the checkpoint to run"
    )

    parser = add_parser_simulation_params(parser)

    parser.add_argument(
        "--num_workers",
        type=int,
        default=NUM_ROLLOUT_WORKERS,
        help="Number of parallel rollout workers.",
    )

    parser.add_argument(
        "--num_tests",
        type=int,
        default=NUM_TESTS,
        help="Number of tests to run for each configuration.",
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default=RESULTS_DIR,
        help="Simulation results output directory name.",
    )

    parser.add_argument(
        "--perturb",
        default=False,
        action="store_true",
        help="Whether to perturb merging vehicles departures.",
    )

    parser.add_argument(
        "--const_control",
        default=False,
        action="store_true",
        help="Whether to use constant control signals (used to get baselines).",
    )

    parser.add_argument(
        "--const_control_val",
        type=float,
        default=None,
        help="Value of control signal to use for the constant control baseline, if used.",
    )

    parser.add_argument(
        "--const_control_val_norm",
        type=float,
        default=None,
        help="Normalized value of control signal to use for the constant control baseline, if used.",
    )
    return parser


def get_sim_configs(sumo_config_params: dict, sim_config_params: dict):
    sim_configs = []
    add_sim_configs(sim_configs, sumo_config_params, sim_config_params)
    return sim_configs


def add_sim_configs(
    sim_configs: list, sumo_config_params: dict, sim_config_params: dict
):
    multiple_sumo_params = [
        key for key, value in sumo_config_params.items() if isinstance(value, list)
    ]
    if len(multiple_sumo_params) > 0:
        sumo_param_name = multiple_sumo_params[0]
        print(f"iterating {sumo_param_name}")
        for val in sumo_config_params[sumo_param_name]:
            add_sim_configs(
                sim_configs,
                sumo_config_params | {sumo_param_name: val},
                sim_config_params,
            )
    else:
        multiple_sim_params = [
            key for key, value in sim_config_params.items() if isinstance(value, list)
        ]
        if len(multiple_sim_params) > 0:
            sim_param_name = multiple_sim_params[0]
            print(f"iterating {sim_param_name}")
            for val in sim_config_params[sim_param_name]:
                add_sim_configs(
                    sim_configs,
                    sumo_config_params,
                    sim_config_params | {sim_param_name: val},
                )
        else:
            if is_valid_params(sumo_config_params | sim_config_params):
                sim_configs.append(
                    dict(
                        sumo_config_params_update=sumo_config_params,
                        sim_config_params=sim_config_params,
                    )
                )


def add_sims_to_queue(sim_queue: queue.Queue, sim_configs: list):
    for sim_config in sim_configs:
        sim_queue.put(sim_config)


def get_const_control_params(env_obj):
    if isinstance(env_obj, SumoEnvCentralizedVel):
        default_val = env_obj._get_control_profile_max_speed()
        min_val = 0
        max_val = default_val
    elif isinstance(env_obj, SumoEnvCentralizedMinGap):
        default_val = env_obj.default_min_gap
        min_val = env_obj.min_min_gap
        max_val = env_obj.max_min_gap
    elif isinstance(env_obj, SumoEnvCentralizedTau):
        default_val = env_obj.default_tau
        min_val = env_obj.min_tau
        max_val = env_obj.max_tau
    else:
        raise ValueError(
            f"The input environment is of an unsupported class: {type(env_obj)}. "
            + f"Please enter an instance of one of the following: {ENV_CLS_STR_OPTIONS}"
        )
    return default_val, min_val, max_val


def is_valid_params(params: dict):
    no_merge = params.get("no_merge") if params.get("no_merge") is not None else False
    use_learned_control = (
        params.get("use_learned_control")
        if params.get("use_learned_control") is not None
        else False
    )
    use_const_control = params.get("use_const_control") or False
    automatic_const_control_duration = (
        params.get("automatic_const_control_duration") or True
    )
    const_control_params = params.get("const_control_params")
    control_types = [use_learned_control, use_const_control]
    if len([control_type for control_type in control_types if control_type]) > 1:
        return False
    if no_merge and use_learned_control:
        return False
    if no_merge and use_const_control and automatic_const_control_duration:
        return False
    if not use_const_control and const_control_params is not None:
        return False
    if use_const_control and const_control_params is None:
        return False

    return True


def simulate(
    sumo_config_params_update: dict,
    sim_config_params: dict,
    worker_index: int | None = None,
    perturb=False,
):
    sumo_config_params = DEF_SUMO_CONFIG | sumo_config_params_update

    merge_flow_duration = (
        sumo_config_params["merge_flow_duration_single_lane"]
        if sumo_config_params["single_lane"]
        else sumo_config_params["merge_flow_duration_multi_lane"]
    )

    seconds_per_step = sim_config_params["seconds_per_step"]

    use_learned_control = sim_config_params["use_learned_control"]
    use_const_control = sim_config_params.get("use_const_control") or False

    if use_const_control and use_learned_control:
        raise ValueError(
            "Only one control scheme can be active at a time. "
            "Please choose either tau or learned control"
        )

    if use_const_control:
        const_control_params = sim_config_params["const_control_params"]
        automatic_const_control_duration = sim_config_params[
            "automatic_const_control_duration"
        ]
        const_control_val = const_control_params.get("const_control_val")
        const_control_val_norm = const_control_params.get("const_control_val_norm")
        if (const_control_val is not None and const_control_val_norm is not None) or (
            const_control_val is None and const_control_val_norm is None
        ):
            raise ValueError(
                "Please provide exactly one of 'const_control_val' or 'const_control_val_norm' in const_control_params"
            )
        const_control_only_rightmost_lane = const_control_params[
            "const_control_only_rightmost_lane"
        ]
        # The following is not relevant if automatic_tau_duration=True
        const_control_start_time = const_control_params["const_control_start_time"]
        const_control_duration = const_control_params["const_control_duration"]
    # RL control params
    if use_learned_control:
        rl_control_params = DEF_RL_CONTROL_PARAMS | sim_config_params.get(
            "rl_control_params", {}
        )

        alg_checkpoint_path = rl_control_params["alg_checkpoint_path"]
        rl_per_lane_control = rl_control_params["rl_per_lane_control"]
        exploit = rl_control_params.get("exploit") or False

    single_lane = sumo_config_params["single_lane"]

    custom_name_postfix = sim_config_params["custom_name_postfix"]
    sim_config_params["per_lane_control"] = False
    if not any([use_learned_control, use_const_control]):
        custom_name_postfix = "no_control"
    elif use_learned_control:
        custom_name_postfix = "rl_control"
        sim_config_params.update(per_lane_control=rl_per_lane_control)
    elif use_const_control:
        custom_name_postfix = (
            f"const_control_{const_control_val}"
            if const_control_val
            else f"const_control_norm_{const_control_val_norm}"
        )
        if const_control_only_rightmost_lane and not single_lane:
            custom_name_postfix += "_rightmost"
            sim_config_params.update(per_lane_control=const_control_only_rightmost_lane)

    random_av_switching = sumo_config_params["random_av_switching"]
    random_av_switching_seed = sumo_config_params["random_av_switching_seed"]

    if random_av_switching and random_av_switching_seed is not None:
        custom_name_postfix += f"_av_switch_seed_{random_av_switching_seed}"

    # rl_episode_id = sim_config_params["rl_episode_id"]
    # if use_learned_control and rl_episode_id is not None:
    #     custom_name_postfix += f"_ep_{rl_episode_id}"

    sim_config_params.update(custom_name_postfix=custom_name_postfix)

    env_config = get_centralized_env_config(
        sumo_config_params_update,
        sim_config_params,
        perturb=perturb,
        perturb_seed=random_av_switching_seed,
    )
    if worker_index is not None:
        env_config = EnvContext(env_config, worker_index=worker_index)

    env_class_obj = get_env_class_from_str(sim_config_params["env_class"])

    # full_state_segments = [
    #     segment
    #     for segment, data in env.segment_data.items()
    #     if data["end"]["edge"] in env.highway_state_edges
    # ]
    # print(f"{full_state_segments = }")
    # print(f"{env.highway_state_segments = }")
    episode_results_dir = get_episode_results_dir(
        env_config.get("results_dir", env_class_obj.DEF_RESULTS_DIR),
        env_config["sumo_config"].sumo_config_file,
        env_config["sumo_config"].scenario_dir,
        env_config["random_av_switching"],
        env_config["av_percent"],
        env_config["name_postfix"],
    )
    if Path(episode_results_dir).exists() and sim_config_params["no_rerun_existing"]:
        directory_entries = os.listdir(episode_results_dir)
        if len(directory_entries) > 1:
            # More than just the metadata file exists in the folder
            print(
                f"results directory {episode_results_dir} already exists. "
                "To rerun existing simulations, specify no_rerun_existing=False"
            )
            return

    # print(f"[{worker_index}] creating env...")
    env = env_class_obj(env_config)
    # print(f"[{worker_index}] created env...")
    # print(f"{env.observation_space = }")
    # print(f"{env.action_space = }")
    # RL control
    policy = None
    if use_learned_control:
        if alg_checkpoint_path is None:
            raise ValueError("Checkpoint path undefined.")
        if not Path(alg_checkpoint_path).exists():
            raise ValueError(f"Checkpoint path {alg_checkpoint_path} does not exist.")
        policy = Policy.from_checkpoint(alg_checkpoint_path)["hierarchical_policy"]
    # Count throughput during merge - initialization
    # TODO: Extract from route file

    starts_with = "DEFAULT_VEHICLE_2." if single_lane else "DEFAULT_VEHICLE_5."

    merging_veh_ids = extract_vehicle_ids_from_routes(env.route_path, starts_with)
    first_veh_id = merging_veh_ids[0] if len(merging_veh_ids) > 0 else ""
    last_veh_id = merging_veh_ids[-1] if len(merging_veh_ids) > 0 else ""
    edge_after_merge = "634155175.210"
    first_merge_veh_exited = False
    last_merge_veh_exited = False
    num_veh_after_merge = 0
    num_merge_timesteps = 0
    prev_veh_in_edge_after_merge = set()

    # Constant control signal:
    first_merge_veh_entered = False

    default_control_val, min_control_val, max_control_val = get_const_control_params(
        env
    )
    if use_const_control:
        norm_default_control_val = (default_control_val - min_control_val) / (
            max_control_val - min_control_val
        )
        norm_const_control_val_action = (
            (
                (const_control_val - min_control_val)
                / (max_control_val - min_control_val)
            )
            if const_control_val is not None
            else const_control_val_norm
        )

    at_least_one_waiting = True
    obs, info = env.reset()
    terminated = {"__all__": False}
    truncated = {"__all__": False}
    episode_reward = 0

    while not (terminated.get("__all__") or truncated.get("__all__")):
        t = env.step_count * env.sumo_config.seconds_per_step

        actions = {}
        if use_learned_control:
            action_norm = policy.compute_single_action(
                obs["centralized"],
                explore=(not exploit),
            )[0]
            action_scaled = unsquash_action(action_norm, env.action_space)
            actions = {env.CENTRALIZED_AGENT_NAME: action_scaled}

        elif use_const_control:
            norm_const_profile = (
                np.ones(env.action_space_len) * norm_default_control_val
            )
            if (
                (first_merge_veh_entered and not last_merge_veh_exited)
                and automatic_const_control_duration
            ) or (
                not automatic_const_control_duration
                and t >= const_control_start_time
                and t <= (const_control_start_time + const_control_duration)
            ):
                if const_control_only_rightmost_lane and not single_lane:
                    rightmost_idx = [0]
                    for cumsum_lanes in env.cumsum_control_segment_lanes[:-1]:
                        rightmost_idx.append(cumsum_lanes)
                    norm_const_profile[rightmost_idx] = norm_const_control_val_action
                else:
                    norm_const_profile[:] = norm_const_control_val_action

            actions = {env.CENTRALIZED_AGENT_NAME: norm_const_profile}

        # Step the environment
        obs, rewards, terminated, truncated, info = env.step(actions)

        # Count throughput during merge
        veh_data = env._get_veh_data()

        first_merge_veh_exited = first_merge_veh_exited or (
            first_veh_id in veh_data.keys()
            and veh_data[first_veh_id][tc.VAR_ROAD_ID] == edge_after_merge
        )

        last_merge_veh_exited = last_merge_veh_exited or (
            last_veh_id in veh_data.keys()
            and veh_data[last_veh_id][tc.VAR_ROAD_ID] == edge_after_merge
        )

        current_veh_in_edge_after_merge = set(
            [
                veh_id
                for veh_id in veh_data.keys()
                if veh_data[veh_id][tc.VAR_ROAD_ID] == edge_after_merge
            ]
        )
        num_new_veh = len(
            current_veh_in_edge_after_merge - prev_veh_in_edge_after_merge
        )
        prev_veh_in_edge_after_merge = current_veh_in_edge_after_merge.copy()

        if first_merge_veh_exited and not last_merge_veh_exited:
            num_veh_after_merge += num_new_veh
            num_merge_timesteps += 1

        first_merge_veh_entered = first_merge_veh_entered or (
            first_veh_id in veh_data.keys()
        )

        episode_reward += sum(rewards.values())
        if env.num_waiting_veh < 1:
            at_least_one_waiting = False

    avg_time_delay = env.veh_travel_info["time_delay"].mean()
    avg_acc_time_delay_per_sec = (
        env.veh_travel_info["time_delay"] / env.veh_travel_info["total_time"]
    ).mean()

    episode_results = dict(
        reward=episode_reward,
        num_completed_veh=env.num_completed_veh,
        num_waiting_veh=env.num_waiting_veh,
        at_least_one_waiting=at_least_one_waiting,
        avg_time_delay=avg_time_delay,
        avg_acc_time_delay_per_sec=avg_acc_time_delay_per_sec,
    )

    if sim_config_params["env_class"] == "SumoEnvCentralizedTau":
        sim_tau = const_control_val if use_const_control else env.default_tau
    else:
        sim_tau = SumoEnvCentralizedTau.DEF_MIN_TAU

    episode_results.update(tau=sim_tau)
    if use_const_control:
        episode_results.update(
            const_control_val=const_control_val, const_control_val_norm=const_control_val_norm, default_control_val=default_control_val
        )

    if num_merge_timesteps > 0:
        merge_time = num_merge_timesteps * seconds_per_step
        merge_avg_outflow = num_veh_after_merge / merge_time * 3600
        max_throughput = 1800 * (1 if single_lane else 4)
        merge_time_efficiency = merge_flow_duration / merge_time
        merge_throughput_efficiency = merge_avg_outflow / max_throughput
        num_merge_veh = len(merging_veh_ids)
        merge_inflow = num_merge_veh / merge_flow_duration * 3600
        avg_veh_merge_time = merge_time / num_merge_veh
        num_veh_lost_per_merge_veh = (
            avg_veh_merge_time * (max_throughput - merge_avg_outflow) / 3600
        )

        episode_results.update(
            num_veh_after_merge=num_veh_after_merge,
            merge_time=merge_time,
            merge_avg_outflow=merge_avg_outflow,
            merge_time_efficiency=merge_time_efficiency,
            merge_throughput_efficiency=merge_throughput_efficiency,
            num_merge_veh=num_merge_veh,
            merge_inflow=merge_inflow,
            avg_veh_merge_time=avg_veh_merge_time,
            num_veh_lost_per_merge_veh=num_veh_lost_per_merge_veh,
        )

    # print(episode_results)

    try:
        # print(f"[{os.getpid()}] writing episode_result.json", flush=True)
        with open(Path(env.episode_results_dir) / "episode_result.json", "w") as fp:
            json.dump(episode_results, fp)
        # print(f"[{os.getpid()}] wrote episode_result.json", flush=True)
    except Exception as e:
        print(f"[{os.getpid()}] json dump raised: {e}", flush=True)

    try:
        # print(f"[{os.getpid()}] calling env.log_episode()", flush=True)
        env.log_episode()
        # print(f"[{os.getpid()}] env.log_episode() returned", flush=True)
    except Exception as e:
        print(f"[{os.getpid()}] env.log_episode() raised: {e}", flush=True)

    try:
        # print(f"[{os.getpid()}] calling analyze_sim_group()", flush=True)
        analyze_sim_group(
            sim_group_dirs={"": env.episode_results_dir},
            save_dir=env.episode_results_dir,
        )
        # print(f"[{os.getpid()}] analyze_sim_group() returned", flush=True)
    except Exception as e:
        print(f"[{os.getpid()}] analyze_sim_group() raised: {e}", flush=True)

    try:
        # print(f"[{os.getpid()}] calling env.close()", flush=True)
        env.close()
        # print(f"[{os.getpid()}] env.close() returned, deleting env...", flush=True)
        del env
        # print(f"[{os.getpid()}] env deleted, collecting garbage...", flush=True)
        gc.collect()
    except Exception as e:
        print(f"[{os.getpid()}] env.close() raised: {e}", flush=True)


def do_job(
    tasks_to_accomplish: queue.Queue,
    worker_index: int | None = None,
    perturb=False,
):
    while True:
        try:
            """
                try to get task from the queue. get_nowait() function will
                raise queue.Empty exception if the queue is empty.
                queue(False) function would do the same task also.
            """
            task = tasks_to_accomplish.get_nowait()
        except queue.Empty:
            break

        simulate(**task, worker_index=worker_index, perturb=perturb)
    return True


def run_all_simulations(
    sumo_config_params,
    sim_config_params,
    num_processes=NUM_ROLLOUT_WORKERS,
    perturb=False,
):
    sim_configs = get_sim_configs(sumo_config_params, sim_config_params)
    # Create shared SUMO config/route files once in the parent process so parallel
    # workers do not race on writing the same files during env initialization.
    for sim_config in sim_configs:
        sumo_params = sim_config["sumo_config_params_update"]
        get_centralized_env_config(
            sumo_params,
            sim_config["sim_config_params"],
            perturb=perturb,
            perturb_seed=(DEF_SUMO_CONFIG | sumo_params)["random_av_switching_seed"],
        )
    sim_queue = multiprocessing.Queue() if num_processes > 0 else queue.Queue()
    add_sims_to_queue(sim_queue, sim_configs)
    # creating processes
    if num_processes > 0:
        processes: list[multiprocessing.Process] = []
        for w in range(num_processes):
            p = multiprocessing.Process(target=do_job, args=[sim_queue, w, perturb])
            processes.append(p)
            p.start()

        # completing process
        for p in processes:
            p.join()

    else:
        print(f"{sim_queue.qsize() = }")
        do_job(sim_queue, perturb=perturb)
        # task = sim_queue.get()
        # simulate(**task)

def main():
    parser = create_eval_parser()
    # Parse the arguments
    args = parser.parse_args()

    num_processes = args.num_workers

    # Set seed to an integer for deterministic simulation. Set to None for
    # default behavior.
    random_seed = args.random_seed
    sumo_seed = random_seed if random_seed is not None else SUMO_SEED  # None
    # random_av_switching_seed = list(np.arange(10) + 20)  # 0 # None
    num_tests = args.num_tests
    random_av_switching_seed = (
        list(np.arange(num_tests))  # 0 # None
    )

    env_class = args.env_class
    alg_checkpoint_path = args.path
    use_learned_control = [False]
    if alg_checkpoint_path is not None:
        use_learned_control.append(True)
    use_const_control = False  # [False, True]
    const_control_only_rightmost_lane = False
    rl_per_lane_control = PER_LANE_CONTROL
    automatic_const_control_duration = True
    num_control_segments = args.num_control_seg

    no_merge = [True, False]
    single_lane = args.single_lane  # [True, False]

    av_percent = args.av_percent  # [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    scenario_dir = Path("scenarios/single_junction") / args.results_dir
    # network_file_name = None
    network_file_name = (
        "short_merge_lane_separate_exit_lane_disconnected_merge_lane.net.xml"
    )

    sumo_config_params = DEF_SUMO_CONFIG_PARAMS | {
        "scenario_dir": scenario_dir,
        "no_merge": no_merge,
        "single_lane": single_lane,
        "av_percent": av_percent,
        "random_av_switching_seed": random_av_switching_seed,
    }

    if network_file_name is not None:
        sumo_config_params.update({"network_file_name": network_file_name})

    const_control_val = (
        np.arange(1.6, 2.6, 0.1).round(2).tolist()
        if not single_lane and not const_control_only_rightmost_lane
        else list(np.arange(2, 6.5, 0.5))
    )

    const_control_constants = {
        "const_control_only_rightmost_lane": const_control_only_rightmost_lane,  # True,
        # The following is not relevant if automatic_const_control_duration=True
        "const_control_start_time": 200,
        "const_control_duration": 100,
    }
    if isinstance(const_control_val, list):
        const_control_params = [None] + [
            const_control_constants | {"const_control_val": control_val}
            for control_val in const_control_val
        ]
    elif use_const_control:
        const_control_params = const_control_constants | {
            "const_control_val": const_control_val
        }
    else:
        const_control_params = None

    if use_learned_control and alg_checkpoint_path is None:
        raise ValueError(
            "To evaluate RL-based control, please provide algorithm checkpoint path"
        )

    rl_control_params = {
        "alg_checkpoint_path": alg_checkpoint_path,
        "rl_per_lane_control": rl_per_lane_control,
    }

    scenario_params = {
        # Single junction
        "scenario_start_edge": "992666043",
        "scenario_end_edge": "634155175.210",
        "highway_state_start_edge": "992666043",
        "highway_state_end_edge": "634155175.210",
        "control_start_edge": "992666042",
        "control_end_edge": "992666042",
        "state_merge_edges": ["277208926"],
    }

    num_simulation_steps_per_step = 5  # 1  # int(5 / (seconds_per_step * 2))  # 40
    simulation_time = 500  # 500 if single_lane else 1000  # 500  # 700  # 1400  # 8400

    env_config_overrides = {
        "flat_obs_space": True,
        # "max_tau": 10,
    }
    custom_name_postfix = None

    sim_config_params = DEF_SIM_CONFIG_PARAMS | {
        "sumo_seed": sumo_seed,
        "use_learned_control": use_learned_control,
        "use_const_control": use_const_control,
        "num_simulation_steps_per_step": num_simulation_steps_per_step,
        "simulation_time": simulation_time,
        "automatic_const_control_duration": automatic_const_control_duration,
        "const_control_params": const_control_params,
        "rl_control_params": rl_control_params,
        "scenario_params": scenario_params,
        "env_config_overrides": env_config_overrides,
        "custom_name_postfix": custom_name_postfix,
        "no_rerun_existing": True,  # False,
        "num_control_segments": num_control_segments,
        "with_eval": True,
        "env_class": env_class,
    }

    run_all_simulations(sumo_config_params, sim_config_params, num_processes)


if __name__ == "__main__":
    main()
