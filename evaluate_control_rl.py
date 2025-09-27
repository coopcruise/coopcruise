# import time
import argparse
import os
import numpy as np

from pathlib import Path

# import multiprocessing
from ray.rllib.algorithms.algorithm import Algorithm, get_checkpoint_info
from sumo_multi_agent_env import SumoConfig

from evaluate_control_new import (
    create_eval_parser,
    SUMO_SEED,
    PER_LANE_CONTROL,
    run_all_simulations,
)
from train_ppo_centralized import (
    DEF_SIM_CONFIG_PARAMS,
    DEF_SUMO_CONFIG_PARAMS,
    NUM_REMOVE_END_STATE_SEGMENTS,
    NUM_REMOVE_START_STATE_SEGMENTS,
    NUM_SIMULATION_STEPS_PER_STEP,
    SIMULATION_TIME,
    SINGLE_LANE,
    AV_PERCENT,
    RANDOM_AV_SWITCHING,
    WARM_UP_TIME,
    NUM_CONTROL_SEGMENTS,
    START_POLICY_AFTER_WARM_UP,
)

MERGE_FLOW_PERCENT = 100
RIGHT_LANE_CONTROL = False
COLOR_AV_BY_ACTION_IDX = False


DEBUG_SUMO_CONFIG_PARAMS = {
    "no_merge": [False],
}

DEBUG_SIM_CONFIG_PARAMS = {
    "use_learned_control": True,
    "use_libsumo": False,
    "no_rerun_existing": False,
    "show_gui_in_traci_mode": True,
}

DEBUG_HUMAN_SIM_CONFIG_PARAMS = DEBUG_SIM_CONFIG_PARAMS | {"use_learned_control": False}

DEBUG_ENV_CONFIG_OVERRIDES = {"color_av_by_action_idx": True}


def get_checkpoint_state(checkpoint_path):
    checkpoint_info = get_checkpoint_info(checkpoint_path)
    state = Algorithm._checkpoint_info_to_algorithm_state(
        checkpoint_info=checkpoint_info
    )
    return state


def get_checkpoint_config(checkpoint_path):
    state = get_checkpoint_state(checkpoint_path)
    return state["config"]


def get_checkpoint_env_config(checkpoint_path):
    config = get_checkpoint_config(checkpoint_path)
    return config["env_config"]


def sumo_config_to_dict(sumo_config: SumoConfig):
    return {attr: getattr(sumo_config, attr) for attr in dir(sumo_config)}


def add_parser_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Whether to run debug mode. In debug mode, SUMO GUI is shown and only a single task is performed",
    )

    parser.add_argument(
        "--debug_human",
        default=False,
        action="store_true",
        help="Whether to run debug mode with human only. In debug mode, SUMO GUI is shown and only a single task is performed",
    )

    parser.add_argument(
        "--auto_results_dir",
        default=False,
        action="store_true",
        help="Whether to use an automatic results directory name based on checkpoint parameters. This will disregard results_dir",
    )
    return parser


if __name__ == "__main__":
    parser = create_eval_parser()
    parser = add_parser_args(parser)
    # Parse the arguments
    args = parser.parse_args()

    debug = args.debug
    debug_human = args.debug_human
    if debug_human:
        debug = True

    num_processes = args.num_workers if not debug else 0

    # Set seed to an integer for deterministic simulation. Set to None for
    # default behavior.
    random_seed = args.random_seed
    sumo_seed = random_seed if random_seed is not None else SUMO_SEED  # None
    # random_av_switching_seed = list(np.arange(10) + 20)  # 0 # None
    num_tests = args.num_tests if not debug else 1

    random_av_switching_seed = (
        list(np.arange(num_tests))  # 0 # None
    )

    alg_checkpoint_path = args.path

    use_tau_control = False  # [False, True]
    tau_control_only_rightmost_lane = False
    automatic_tau_duration = True

    assert alg_checkpoint_path is not None, "please provide algorithm checkpoint path"
    assert os.path.exists(alg_checkpoint_path), (
        f"checkpoint path does not exist: {alg_checkpoint_path}"
    )

    no_merge = [False, True]
    use_learned_control = [False, True]
    no_rerun_existing = True
    color_av_by_action_idx = COLOR_AV_BY_ACTION_IDX

    # Extract from checkpoint
    env_class = get_checkpoint_config(alg_checkpoint_path)["env"]
    env_config = get_checkpoint_env_config(alg_checkpoint_path)
    rl_per_lane_control = env_config.get("per_lane_control") or PER_LANE_CONTROL
    num_control_segments = (
        env_config.get("num_control_segments") or NUM_CONTROL_SEGMENTS
    )
    single_lane = env_config.get("is_single_lane") or SINGLE_LANE
    av_percent = env_config.get("av_percent") or AV_PERCENT
    random_av_switching = env_config.get("random_av_switching") or RANDOM_AV_SWITCHING
    warm_up_time = env_config.get("warm_up_time") or WARM_UP_TIME
    right_lane_control = env_config.get("right_lane_control") or RIGHT_LANE_CONTROL
    start_policy_after_warm_up = (
        env_config.get("start_policy_after_warm_up") or START_POLICY_AFTER_WARM_UP
    )
    num_remove_start_state_segments = (
        env_config.get("num_remove_start_state_segments")
        or NUM_REMOVE_START_STATE_SEGMENTS
    )
    num_remove_end_state_segments = (
        env_config.get("num_remove_end_state_segments") or NUM_REMOVE_END_STATE_SEGMENTS
    )

    sumo_config_file_name: str = env_config["sumo_config"].sumo_config_file
    if "merge_flow_percent" in sumo_config_file_name:
        merge_flow_percent = int(
            sumo_config_file_name.split("merge_flow_percent_")[-1].split("_")[0]
        )
    else:
        merge_flow_percent = MERGE_FLOW_PERCENT

    num_simulation_steps_per_step = (
        env_config.get("num_simulation_steps_per_step") or NUM_SIMULATION_STEPS_PER_STEP
    )
    simulation_time = env_config.get("simulation_time") or SIMULATION_TIME

    # network_file_name = None
    network_file_name = (
        "short_merge_lane_separate_exit_lane_disconnected_merge_lane.net.xml"
    )

    results_dir_name = args.results_dir
    if args.auto_results_dir:
        lane_type_str = "_single_lane" if single_lane else "_multi_lane"
        control_type_str = (
            "_right_lane_control"
            if right_lane_control
            else ("_per_lane_control" if rl_per_lane_control else "")
        )
        merge_inflow_str = f"_merge_flow_percent_{merge_flow_percent}"
        results_dir_name = (
            f"{env_class}{control_type_str}{merge_inflow_str}{lane_type_str}"
        )

    scenario_dir = Path("scenarios/single_junction") / results_dir_name

    sumo_config_params = DEF_SUMO_CONFIG_PARAMS | {
        "scenario_dir": scenario_dir,
        "od_flow_file_name": "edge_flows_interval_8400_taz_reduced",
        "no_merge": no_merge,
        "single_lane": single_lane,
        "av_percent": av_percent,
        "warm_up_time": warm_up_time,
        "merge_flow_percent": merge_flow_percent,
        "random_av_switching": random_av_switching,
        "random_av_switching_seed": random_av_switching_seed,
    }

    if network_file_name is not None:
        sumo_config_params.update({"network_file_name": network_file_name})

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

    env_config_overrides = {
        "flat_obs_space": True,
        "right_lane_control": right_lane_control,
        "color_av_by_action_idx": color_av_by_action_idx,
        "start_policy_after_warm_up": start_policy_after_warm_up,
        "num_remove_start_state_segments": num_remove_start_state_segments,
        "num_remove_end_state_segments": num_remove_end_state_segments,
    }
    custom_name_postfix = None

    sim_config_params = DEF_SIM_CONFIG_PARAMS | {
        "sumo_seed": sumo_seed,
        "use_learned_control": use_learned_control,
        "use_tau_control": False,
        "num_simulation_steps_per_step": num_simulation_steps_per_step,
        "simulation_time": simulation_time,
        "automatic_tau_duration": True,
        "tau_control_params": None,
        "rl_control_params": rl_control_params,
        "scenario_params": scenario_params,
        "env_config_overrides": env_config_overrides,
        "custom_name_postfix": custom_name_postfix,
        "no_rerun_existing": no_rerun_existing,
        "num_control_segments": num_control_segments,
        "with_eval": True,
        "env_class": env_class,
    }

    if debug or debug_human:
        sumo_config_params |= DEBUG_SUMO_CONFIG_PARAMS
        sim_config_params |= (
            DEBUG_HUMAN_SIM_CONFIG_PARAMS if debug_human else DEBUG_SIM_CONFIG_PARAMS
        )
        env_config_overrides |= DEBUG_ENV_CONFIG_OVERRIDES

    run_all_simulations(sumo_config_params, sim_config_params, num_processes)
