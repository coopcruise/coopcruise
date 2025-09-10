from pathlib import Path

from sumo_multi_agent_env import SumoConfig, EvaluationConfig

from utils.sumo_utils import (
    extract_highway_profile_detector_file_name,
)
from utils.sumo_config_creation_pipeline import (
    get_sumo_config_od_file_stem,
    get_sumo_config_output_suffix,
    get_sumo_config_creation_params,
    create_sumo_config,
)
from utils.i24_utils import get_main_road_west_edges


DEF_SUMO_CONFIG = {
    "scenario_dir": "scenarios/single_junction/test_calibrated",
    "od_flow_file_name": "edge_flows_interval_8400_taz_reduced",
    "network_file_name": "short_merge_lane_separate_exit_lane.net.xml",
    "no_merge": False,
    "single_lane": False,
    "change_lc_av_only": False,
    "no_lc": False,
    "no_lc_right": False,
    "lc_params": {},
    "av_percent": 100,
    "warm_up_time": 200,
    "merge_flow_duration_single_lane": 30,
    "merge_flow_duration_multi_lane": 50,
    "merge_flow_percent": 100,
    "break_period_duration": 8400,
    "default_tau": None,
    "keep_veh_names_no_merge": True,
    "inflow_time_headway": 2,
    "human_speed_std_0": False,
    "no_rerun_existing": True,
    "random_av_switching": False,
    "random_av_switching_seed": None,
}
DEF_ENV_CONFIG_PARAMS = {
    "normalize_car_following_obs": False,
    "use_outflow_reward": False,
    "use_time_delay_reward": True,
    "include_tse_pos_in_obs": False,
    "include_av_frac_in_obs": False,
    "flat_obs_space": False,
}


def get_sumo_config_file_name(config_overrides: dict):
    config = DEF_SUMO_CONFIG | config_overrides

    scenario_dir: str | Path = config["scenario_dir"]
    no_merge: bool = config["no_merge"]
    single_lane: bool = config["single_lane"]
    av_percent: int = config["av_percent"] if not config["random_av_switching"] else 0
    warm_up_time: int = config["warm_up_time"]
    merge_flow_duration_single_lane: int = config["merge_flow_duration_single_lane"]
    merge_flow_duration_multi_lane: int = config["merge_flow_duration_multi_lane"]
    break_period_duration: int = config["break_period_duration"]
    keep_veh_names_no_merge: bool = config["keep_veh_names_no_merge"]

    merge_flow_duration = (
        merge_flow_duration_single_lane
        if single_lane
        else merge_flow_duration_multi_lane
    )

    sumo_config_file_name = (
        get_sumo_config_od_file_stem(config)
        + (
            f"_periodic_warmup_{warm_up_time}"
            f"_high_{merge_flow_duration}s"
            f"_low_{break_period_duration}s"
            if not no_merge or keep_veh_names_no_merge
            else ""
        )
        + ("_single_lane" if single_lane else "")
        + get_sumo_config_output_suffix(config)
        + (f"_av_{av_percent}_percent" if av_percent >= 0 else "")
        + ".sumocfg"
    )
    return sumo_config_file_name, Path(scenario_dir) / sumo_config_file_name


def create_missing_sumo_config_files(sumo_config_params_update: dict):
    sumo_config_params = DEF_SUMO_CONFIG | sumo_config_params_update
    merge_flow_duration_single_lane = sumo_config_params[
        "merge_flow_duration_single_lane"
    ]
    merge_flow_duration_multi_lane = sumo_config_params[
        "merge_flow_duration_multi_lane"
    ]
    merge_flow_duration = (
        merge_flow_duration_single_lane
        if sumo_config_params["single_lane"]
        else merge_flow_duration_multi_lane
    )

    simplified_config_overrides = {
        "od_flow_file_name": sumo_config_params["od_flow_file_name"],
        "network_file_name": sumo_config_params["network_file_name"],
        "output_dir": sumo_config_params["scenario_dir"],
        "single_lane": sumo_config_params["single_lane"],
        "av_penetration_percentage": (
            sumo_config_params["av_percent"]
            if not sumo_config_params["random_av_switching"]
            else 0
        ),
        "no_merge": sumo_config_params["no_merge"],
        "keep_veh_names_no_merge": sumo_config_params["keep_veh_names_no_merge"],
        "default_tau": sumo_config_params["default_tau"],
        "inflow_time_headway": sumo_config_params["inflow_time_headway"],
        "change_lc_av_only": sumo_config_params["change_lc_av_only"],
        "no_lc": sumo_config_params["no_lc"],
        "no_lc_right": sumo_config_params["no_lc_right"],
        "custom_av_lc": sumo_config_params["lc_params"] is not None,
        "av_lc": sumo_config_params["lc_params"],
        "warm_up_time": sumo_config_params["warm_up_time"],
        "merge_flow_duration": merge_flow_duration,
        "merge_flow_percent": sumo_config_params["merge_flow_percent"],
        "break_period_duration": sumo_config_params["break_period_duration"],
        "human_speed_std_0": sumo_config_params["human_speed_std_0"],
    }
    sumo_creation_config = get_sumo_config_creation_params(simplified_config_overrides)
    sumo_config_path = create_sumo_config(sumo_creation_config)
    return sumo_config_path


def get_centralized_env_config(
    sumo_config_params_update: dict, sim_config_params: dict
):
    sumo_config_params = DEF_SUMO_CONFIG | sumo_config_params_update
    sumo_config_file_name, sumo_config_path = get_sumo_config_file_name(
        sumo_config_params
    )

    if not Path(sumo_config_path).exists():
        sumo_config_path = create_missing_sumo_config_files(sumo_config_params_update)

    use_libsumo = sim_config_params["use_libsumo"]
    show_gui_in_traci_mode = sim_config_params["show_gui_in_traci_mode"]
    sumo_seed = sim_config_params["sumo_seed"]

    seconds_per_step = sim_config_params["seconds_per_step"]

    num_control_segments = (
        sim_config_params["num_control_segments"]
        if sim_config_params["num_control_segments"] is not None
        else 0
    )

    speed_profile_detector_file = extract_highway_profile_detector_file_name(
        sumo_config_path
    )

    sumo_config_input = dict(
        scenario_dir=sumo_config_params["scenario_dir"],
        sumo_config_file=sumo_config_file_name,
        seconds_per_step=seconds_per_step,
        show_gui=show_gui_in_traci_mode,
        speed_profile_detector_file=speed_profile_detector_file,
        no_warnings=True,
        seed=sumo_seed,
    )

    if use_libsumo:
        sumo_config_input.update(dict(show_gui=False, use_libsumo=True))

    sumo_config = SumoConfig(**sumo_config_input)

    eval_config = EvaluationConfig()

    single_lane = sumo_config_params["single_lane"]
    num_simulation_steps_per_step = sim_config_params["num_simulation_steps_per_step"]
    simulation_time = sim_config_params["simulation_time"]
    warm_up_time = sumo_config_params["warm_up_time"]  # 150  # 240  # 1200

    # Scenario
    scenario_params = sim_config_params["scenario_params"]
    scenario_start_edge = scenario_params["scenario_start_edge"]
    scenario_end_edge = scenario_params["scenario_end_edge"]
    highway_state_start_edge = scenario_params["highway_state_start_edge"]
    highway_state_end_edge = scenario_params["highway_state_end_edge"]
    control_start_edge = scenario_params["control_start_edge"]
    control_end_edge = scenario_params["control_end_edge"]

    with_eval = sim_config_params["with_eval"]
    eval_config = EvaluationConfig() if with_eval else None

    highway_edges = get_main_road_west_edges(with_internal=False)

    scenario_edges = highway_edges[
        highway_edges.index(scenario_start_edge) : highway_edges.index(
            scenario_end_edge
        )
        + 1
    ]

    highway_state_edges = highway_edges[
        highway_edges.index(highway_state_start_edge) : highway_edges.index(
            highway_state_end_edge
        )
        + 1
    ]

    control_edges_list = highway_edges[
        highway_edges.index(control_start_edge) : highway_edges.index(control_end_edge)
        + 1
    ]
    control_edges = {edge: {"start": 0, "end": -1} for edge in control_edges_list}

    env_config_params = (
        DEF_ENV_CONFIG_PARAMS | sim_config_params["env_config_overrides"]
    )

    env_config = dict(
        num_simulation_steps_per_step=num_simulation_steps_per_step,
        simulation_time=simulation_time,
        sumo_config=sumo_config,
        warm_up_time=warm_up_time,
        control_edges=control_edges,
        num_control_segments=num_control_segments,
        eval_config=eval_config,
        highway_sorted_road_edges=scenario_edges,
        highway_state_edges=highway_state_edges,
        state_merge_edges=scenario_params["state_merge_edges"],
        is_single_lane=single_lane,
        random_av_switching=sumo_config_params["random_av_switching"],
        av_percent=sumo_config_params["av_percent"],
        random_av_switching_seed=sumo_config_params["random_av_switching_seed"],
        per_lane_control=sim_config_params["per_lane_control"],
        name_postfix=sim_config_params["custom_name_postfix"],
        **env_config_params,
    )

    return env_config
