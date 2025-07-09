import json
import argparse

# import time
import numpy as np

from pathlib import Path

# import multiprocessing
from multiprocessing import Process, Queue
import queue  # imported for using queue.Empty exception
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.spaces.space_utils import unsquash_action
from ray.rllib.env.env_context import EnvContext
import traci.constants as tc

from sumo_centralized_envs_new import SumoEnvCentralizedTau, SumoEnvCentralizedVel
from utils.sumo_utils import extract_vehicle_ids_from_routes
from utils.sim_utils import DEF_SUMO_CONFIG, get_tau_env_config

NUM_ROLLOUT_WORKERS = 10
INFLOW_TIME_HEADWAY = 2
AV_PERCENT = 100
ENV_CLS = "SumoEnvCentralizedTau"
ENV_CLS_OPTIONS = ["SumoEnvCentralizedTau", "SumoEnvCentralizedVel"]

SINGLE_LANE = False  # True
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
RANDOM_AV_SWITCHING_SEED = None  # 0  # None

NUM_TESTS = 30

RESULTS_DIR = "test"

CHANGE_LC_AV_ONLY = False
NO_LC = False
NO_LC_RIGHT = True
LC_PARAMS = (
    None  # dict(lcKeepRight=0, lcAssertive=2.5, lcSpeedGain=5, lcImpatience=0.7)
)
DEFAULT_TAU = None
RANDOM_AV_SWITCHING = True
HUMAN_SPEED_STD_0 = True

WARM_UP_TIME = 200  # sec
MERGE_FLOW_DURATION_SINGLE_LANE = 30  # sec
MERGE_FLOW_DURATION_MULTI_LANE = 50  # sec
BREAK_PERIOD_DURATION = 8400  # sec

KEEP_VEH_NAMES_NO_MERGE = True

# NETWORK_FILE_NAME = None
NETWORK_FILE_NAME = (
    "short_merge_lane_separate_exit_lane_disconnected_merge_lane.net.xml"
)

DEF_RL_CONTROL_PARAMS = {
    "alg_checkpoint_path": None,  # Must be overridden to use RL control
    "rl_per_lane_control": False,
}

DEF_TAU_CONTROL_CONSTANTS = {
    "tau_control_only_rightmost_lane": True,
    # The following is not relevant if automatic_tau_duration=True
    "tau_control_start_time": 200,
    "tau_control_duration": 100,
}


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Train area-based time-headway traffic congestion controller using PPO.",
        epilog="python3 -i <this-script>",
    )

    parser.add_argument(
        "path", type=str, nargs="?", default=None, help="Path to the checkpoint to run"
    )

    # optional input parameters
    parser.add_argument(
        "--env_class",
        type=str,
        default=ENV_CLS,
        help=f"Environment class name for evaluation. One of: {ENV_CLS_OPTIONS}",
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


def add_sims_to_queue(sim_queue: Queue, sim_configs: list):
    for sim_config in sim_configs:
        sim_queue.put(sim_config)


def is_valid_params(params: dict):
    no_merge = params.get("no_merge") if params.get("no_merge") is not None else False
    use_learned_control = (
        params.get("use_learned_control")
        if params.get("use_learned_control") is not None
        else False
    )
    use_tau_control = (
        params.get("use_tau_control")
        if params.get("use_tau_control") is not None
        else False
    )
    automatic_tau_duration = (
        params.get("automatic_tau_duration")
        if params.get("automatic_tau_duration") is not None
        else True
    )
    tau_control_params = params.get("tau_control_params")
    use_vsl_control = (
        params.get("use_vsl_control")
        if params.get("use_vsl_control") is not None
        else False
    )
    control_types = [use_learned_control, use_tau_control, use_vsl_control]
    if len([control_type for control_type in control_types if control_type]) > 1:
        return False
    if no_merge and use_learned_control:
        return False
    if no_merge and use_tau_control and automatic_tau_duration:
        return False
    if not use_tau_control and tau_control_params is not None:
        return False
    if use_tau_control and tau_control_params is None:
        return False

    return True


def simulate(
    sumo_config_params_update: dict,
    sim_config_params: dict,
    worker_index: int | None = None,
):
    sumo_config_params = DEF_SUMO_CONFIG | sumo_config_params_update

    merge_flow_duration = (
        sumo_config_params["merge_flow_duration_single_lane"]
        if sumo_config_params["single_lane"]
        else sumo_config_params["merge_flow_duration_multi_lane"]
    )

    seconds_per_step = sim_config_params["seconds_per_step"]

    use_learned_control = sim_config_params["use_learned_control"]
    use_tau_control = sim_config_params["use_tau_control"]

    if use_tau_control and use_learned_control:
        raise ValueError(
            "Only one control scheme can be active at a time. "
            "Please choose either tau or learned control"
        )

    if use_tau_control:
        tau_control_params = sim_config_params["tau_control_params"]
        automatic_tau_duration = sim_config_params["automatic_tau_duration"]
        tau_val = tau_control_params["tau_val"]
        tau_control_only_rightmost_lane = tau_control_params[
            "tau_control_only_rightmost_lane"
        ]
        # The following is not relevant if automatic_tau_duration=True
        tau_control_start_time = tau_control_params["tau_control_start_time"]
        tau_control_duration = tau_control_params["tau_control_duration"]
    # RL control params
    if use_learned_control:
        rl_control_params = DEF_RL_CONTROL_PARAMS | sim_config_params.get(
            "rl_control_params", {}
        )

        alg_checkpoint_path = rl_control_params["alg_checkpoint_path"]
        rl_per_lane_control = rl_control_params["rl_per_lane_control"]

    single_lane = sumo_config_params["single_lane"]

    custom_name_postfix = sim_config_params["custom_name_postfix"]
    sim_config_params["per_lane_control"] = False
    if not any([use_learned_control, use_tau_control]):
        custom_name_postfix = "no_control"
    elif use_learned_control:
        custom_name_postfix = "rl_control"
        sim_config_params.update(per_lane_control=rl_per_lane_control)
    elif use_tau_control:
        custom_name_postfix = f"tau_control_{tau_val}"
        if tau_control_only_rightmost_lane and not single_lane:
            custom_name_postfix += "_rightmost"
            sim_config_params.update(per_lane_control=tau_control_only_rightmost_lane)

    random_av_switching = sumo_config_params["random_av_switching"]
    random_av_switching_seed = sumo_config_params["random_av_switching_seed"]

    if random_av_switching and random_av_switching_seed is not None:
        custom_name_postfix += f"_av_switch_seed_{random_av_switching_seed}"

    # rl_episode_id = sim_config_params["rl_episode_id"]
    # if use_learned_control and rl_episode_id is not None:
    #     custom_name_postfix += f"_ep_{rl_episode_id}"

    sim_config_params.update(custom_name_postfix=custom_name_postfix)

    env_config = get_tau_env_config(sumo_config_params_update, sim_config_params)
    if worker_index is not None:
        env_config = EnvContext(env_config, worker_index=worker_index)

    if sim_config_params["env_class"] == "SumoEnvCentralizedTau":
        env_class_obj = SumoEnvCentralizedTau
    elif sim_config_params["env_class"] == "SumoEnvCentralizedVel":
        env_class_obj = SumoEnvCentralizedVel
    else:
        raise ValueError(f"env_class argument must be one of: {ENV_CLS_OPTIONS}")

    env = env_class_obj(env_config)

    if (
        Path(env.episode_results_dir).exists()
        and sim_config_params["no_rerun_existing"]
    ):
        env.close()
        print(
            f"results directory {env.episode_results_dir} already exists. "
            "To rerun existing simulations, specify no_rerun_existing=False"
        )
        return

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

    # Tau control:
    first_merge_veh_entered = False
    if use_tau_control:
        norm_default_tau = (env.default_tau - env.min_tau) / (env.max_tau - env.min_tau)
        norm_tau_val = (tau_val - env.min_tau) / (env.max_tau - env.min_tau)

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
                obs["centralized"], explore=True
            )[0]
            action_scaled = unsquash_action(action_norm, env.action_space)
            actions = {env.CENTRALIZED_AGENT_NAME: action_scaled}

        elif use_tau_control:
            norm_tau_profile = np.ones(env.action_space_len) * norm_default_tau
            if (
                (first_merge_veh_entered and not last_merge_veh_exited)
                and automatic_tau_duration
            ) or (
                t >= tau_control_start_time
                and t <= (tau_control_start_time + tau_control_duration)
                and not automatic_tau_duration
            ):
                if tau_control_only_rightmost_lane and not single_lane:
                    rightmost_idx = [0]
                    for cumsum_lanes in env.cumsum_control_segment_lanes[:-1]:
                        rightmost_idx.append(cumsum_lanes)
                    norm_tau_profile[rightmost_idx] = norm_tau_val
                else:
                    norm_tau_profile[:] = norm_tau_val

            actions = {env.CENTRALIZED_AGENT_NAME: norm_tau_profile}

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
        tau=(tau_val if use_tau_control else env.default_tau),
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

    print(episode_results)

    with open(Path(env.episode_results_dir) / "episode_result.json", "w") as fp:
        json.dump(episode_results, fp)

    env.log_episode()
    env.close()


def do_job(tasks_to_accomplish: Queue, worker_index: int | None = None):
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

        simulate(**task, worker_index=worker_index)
    return True


if __name__ == "__main__":
    parser = create_parser()
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
    use_tau_control = False  # [False, True]
    tau_control_only_rightmost_lane = False
    rl_per_lane_control = PER_LANE_CONTROL
    automatic_tau_duration = True
    num_control_segments = args.num_control_seg

    no_merge = [True, False]
    single_lane = args.single_lane  # [True, False]

    av_percent = args.av_percent  # [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    scenario_dir = Path("scenarios/single_junction") / args.results_dir
    # network_file_name = None
    network_file_name = (
        "short_merge_lane_separate_exit_lane_disconnected_merge_lane.net.xml"
    )

    sumo_config_params = {
        "scenario_dir": scenario_dir,
        "od_flow_file_name": "edge_flows_interval_8400_taz_reduced",
        "no_merge": no_merge,
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

    if network_file_name is not None:
        sumo_config_params.update({"network_file_name": network_file_name})

    tau_val = (
        np.arange(1.6, 2.6, 0.1).round(2).tolist()
        if not single_lane and not tau_control_only_rightmost_lane
        else list(np.arange(2, 6.5, 0.5))
    )

    tau_control_constants = {
        "tau_control_only_rightmost_lane": tau_control_only_rightmost_lane,  # True,
        # The following is not relevant if automatic_tau_duration=True
        "tau_control_start_time": 200,
        "tau_control_duration": 100,
    }
    if isinstance(tau_val, list):
        tau_control_params = [None] + [
            tau_control_constants | {"tau_val": tau} for tau in tau_val
        ]
    elif use_tau_control:
        tau_control_params = tau_control_constants | {"tau_val": tau_val}
    else:
        tau_control_params = None

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

    sim_config_params = {
        "use_libsumo": USE_LIBSUMO,
        "show_gui_in_traci_mode": SHOW_GUI_IN_TRACI_MODE,
        "sumo_seed": sumo_seed,
        "seconds_per_step": 0.5,
        "use_learned_control": use_learned_control,
        "use_tau_control": use_tau_control,
        "num_simulation_steps_per_step": num_simulation_steps_per_step,
        "simulation_time": simulation_time,
        "automatic_tau_duration": automatic_tau_duration,
        "tau_control_params": tau_control_params,
        "rl_control_params": rl_control_params,
        "scenario_params": scenario_params,
        "env_config_overrides": env_config_overrides,
        "custom_name_postfix": custom_name_postfix,
        "no_rerun_existing": True,  # False,
        "num_control_segments": num_control_segments,
        "with_eval": True,
        "env_class": env_class,
    }

    sim_queue = Queue()
    sim_configs = get_sim_configs(sumo_config_params, sim_config_params)
    add_sims_to_queue(sim_queue, sim_configs)
    # creating processes
    if num_processes > 0:
        processes: list[Process] = []
        for w in range(num_processes):
            p = Process(target=do_job, args=[sim_queue, w])
            processes.append(p)
            p.start()

        # completing process
        for p in processes:
            p.join()

    else:
        print(f"{len(sim_queue._buffer) = }")
        do_job(sim_queue)
        # task = sim_queue.get()
        # simulate(**task)
