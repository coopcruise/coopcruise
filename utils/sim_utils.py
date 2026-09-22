from pathlib import Path
import random
from typing import Optional
import xml.etree.ElementTree as ET

from sumo_multi_agent_env import SumoConfig, EvaluationConfig

from utils.metrics_utils import save_xml_element
from utils.sumo_utils import (
    extract_highway_profile_detector_file_name,
    get_route_file_path,
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
    "network_file_name": "short_merge_lane_separate_exit_lane_disconnected_merge_lane.net.xml",
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
    "inflow_percent": 100,
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
        "inflow_percent": sumo_config_params["inflow_percent"],
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
    sumo_config_params_update: dict,
    sim_config_params: dict,
    perturb: bool = False,
    perturb_seed: int = 0,
):
    sumo_config_params = DEF_SUMO_CONFIG | sumo_config_params_update
    sumo_config_file_name, sumo_config_path = get_sumo_config_file_name(
        sumo_config_params
    )

    if not Path(sumo_config_path).exists():
        sumo_config_path = create_missing_sumo_config_files(sumo_config_params_update)

    if perturb:
        perturb_sumo_config_file_name, perturb_sumo_config_path = (
            get_perturbed_sumo_config_file_name(
                sumo_config_file_name, sumo_config_path, perturb_seed=perturb_seed
            )
        )
        print(f"{perturb_sumo_config_file_name = }")
        print(f"{perturb_sumo_config_path = }")
        if not Path(perturb_sumo_config_path).exists():
            create_perturbed_sumo_config_files(
                perturb_sumo_config_path,
                sumo_config_path,
                perturb_seed=perturb_seed,
            )

        warm_up_time_perturbed = get_warm_up_from_sumo_config(perturb_sumo_config_path)
        if warm_up_time_perturbed is not None:
            sumo_config_params["warm_up_time"] = warm_up_time_perturbed

        sumo_config_file_name = perturb_sumo_config_file_name

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


def get_perturbed_sumo_config_file_name(
    sumo_config_file_name: str, sumo_config_path: str | Path, perturb_seed: int = 0
):
    sumo_config_file_name_perturb = (
        sumo_config_file_name.strip(".sumocfg")
        + f"_perturb_seed_{perturb_seed}"
        + ".sumocfg"
    )
    return sumo_config_file_name_perturb, Path(
        sumo_config_path
    ).parent / sumo_config_file_name_perturb


def create_perturbed_sumo_config_files(
    perturb_sumo_config_path: str | Path,
    sumo_config_path: str | Path,
    perturb_seed: int = 0,
):
    assert Path(sumo_config_path).exists(), (
        f"SUMO config file not found in: {sumo_config_path}"
    )
    original_route_file_path = get_route_file_path(sumo_config_path)
    route_file_end = ".rou.xml"
    output_route_file_path = Path(original_route_file_path).parent / (
        original_route_file_path.name.strip(route_file_end)
        + f"_perturb_seed_{perturb_seed}"
        + route_file_end
    )
    perturb_departure_times(
        original_route_file_path,
        output_route_file_path,
        # TODO: Make departure_taz value more general
        departure_taz="taz_4",
        seed=perturb_seed,
    )
    change_sumo_config_route_file(
        sumo_config_path=sumo_config_path,
        output_path=perturb_sumo_config_path,
        old_route_file_name=Path(original_route_file_path).name,
        new_route_file_name=Path(output_route_file_path).name,
    )


def perturb_departure_times(
    input_route_file_path: str,
    output_route_file_path: str,
    departure_taz: str,
    seed: int | None = None,
):
    """
    Perturb departure times of vehicles departing from departure_taz
    by first applying a global constant offset, then
    applying per-vehicle offsets.

    Parameters
    ----------
    input_xml : str
        Path to input routes XML file.
    output_xml : str
        Path to output modified routes XML file.
    seed : int
        Random seed for reproducibility.
    """
    random.seed(seed)

    tree = ET.parse(input_route_file_path)
    root = tree.getroot()

    # Step 1: sample a single constant offset for all relevant vehicles
    constant_offset = round(random.uniform(-1.0, 1.0), 2)
    print(f"Global constant offset: {constant_offset} s")

    count = 0
    for veh in root.findall("vehicle"):
        if veh.get("fromTaz") == departure_taz:
            depart = float(veh.get("depart"))

            # Step 2: sample an individual offset for this vehicle
            per_vehicle_offset = round(random.uniform(-0.2, 0.2), 2)

            # Apply both offsets
            new_depart = depart + constant_offset + per_vehicle_offset
            # Optionally clamp to zero to avoid negative times
            new_depart = max(0.0, new_depart)

            veh.set("depart", f"{new_depart}")
            count += 1

    print(f"Updated {count} vehicles departing from {departure_taz}")

    Path(output_route_file_path).parent.mkdir(parents=True, exist_ok=True)
    save_xml_element(
        root, output_route_file_path, encoding="utf-8", xml_declaration=True
    )


def change_sumo_config_route_file(
    sumo_config_path: str,
    output_path: str,
    old_route_file_name: str,
    new_route_file_name: str,
):
    """
    Update the value of route-files in a configuration XML file.

    Parameters
    ----------
    config_xml : str
        Path to the original configuration XML.
    output_config_xml : str
        Path to save the updated configuration XML.
    old_route : str
        The original route file path as appears in the config.
    new_route : str
        The new route file path to replace it with.
    """
    tree = ET.parse(sumo_config_path)
    root = tree.getroot()

    # Find route-files element
    route_elem = root.find("./input/route-files")
    if route_elem is None:
        raise ValueError("Could not find <route-files> element in configuration XML.")

    current_value = route_elem.get("value")
    print(f"Original route-files value: {current_value}")

    # Replace it
    if current_value == old_route_file_name:
        route_elem.set("value", new_route_file_name)
    else:
        print(
            "Warning: original route file name in config does not match the provided old_route."
        )
        print("Replacing anyway.")
        route_elem.set("value", new_route_file_name)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"Updated configuration saved to: {output_path}")


def get_warm_up_from_sumo_config(sumo_config_path):
    assert Path(sumo_config_path).exists(), (
        f"SUMO config file not found in: {sumo_config_path}"
    )
    route_file_path = get_route_file_path(sumo_config_path)
    # TODO: Make departure_taz value more general
    warm_up_time = earliest_depart_from_taz(route_file_path, departure_taz="taz_4")
    return warm_up_time


def earliest_depart_from_taz(
    route_xml_path: str, departure_taz: str
) -> Optional[float]:
    """
    Finds the earliest departure time of vehicles departing from taz_4.

    Parameters
    ----------
    route_xml_path : str
        Path to the routes XML file.

    Returns
    -------
    float or None
        The earliest departure time among vehicles with fromTaz=departure_taz.
        Returns None if no such vehicles are found.
    """
    tree = ET.parse(route_xml_path)
    root = tree.getroot()

    earliest_time = None

    for veh in root.findall("vehicle"):
        if veh.get("fromTaz") == departure_taz:
            depart_time = float(veh.get("depart"))
            if earliest_time is None or depart_time < earliest_time:
                earliest_time = depart_time

    return earliest_time
