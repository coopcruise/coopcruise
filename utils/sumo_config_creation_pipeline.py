from copy import deepcopy
from pathlib import Path
from utils.create_periodical_inflow_od_matrix import create_and_save_periodical_od
from utils.od_matrix_to_flow_routes import create_and_save_sumo_config_file
from utils.add_avs_to_routes import add_avs_to_routes
from utils.create_detectors_for_highway_profile import (
    create_and_save_detectors_for_highway_profile,
    I24_HIGHWAY_WESTBOUND_EDGES,
)

DEF_VEHICLE_PARAMS = {
    "id": "DEFAULT_VEHTYPE",
    "color": "grey",
    "carFollowModel": "IDM",
    "lcSpeedGain": "1",  # "200",
    "lcStrategic": "20.0",  # "1.0",
    "lcAssertive": "5.0",
    "lcCooperative": "1",
    # "speedFactor": "1.25",
    "speedFactor": "1.0",
    "speedDev": "0.05",
    # Desired headway. SUMO default: 1.0
    # "tau": "1.0",
    "tau": "1.5",
    # "tau": "2.0",
}

DEF_AV_PARAMS = {
    "id": "AV",
    "color": "green",
    "speedDev": "0",
    # "sigma": "0",
}

DEF_SINGLE_LANE_VEHICLE_PARAMS = {
    "lcSpeedGain": "0",  # "200",
    "lcStrategic": "0",  # "1.0",
    "lcAssertive": "20.0",
    "lcCooperative": "-1",
}

DEF_SEGMENT_CONFIG = {
    "edges": I24_HIGHWAY_WESTBOUND_EDGES,
    "detector_results_output_file_name": "speed_profile_detector_results.xml",
    "segment_nominal_length": 100,  # m
    "update_time": 60,  # seconds
    "use_internal_edges": False,
}

DEF_PERIODIC_FLOW_CONFIG = {
    # period_time of a single number will create equal duration high and low
    # periods. To create unequal durations, period_time should be a tuple of
    # (high_duration, low_duration).
    "period_time": 8400,
    # od_oscillations define the scaling factor of flow during high periods and
    # low periods for each required origin and destination pair. For example,
    # ("taz_4", "taz_reduced_end") are the origin and destination of merging
    # flow in exit 59 on highway i-24.
    "od_oscillations": {
        ("taz_4", "taz_reduced_end"): {"high_scale": 1, "low_scale": 0}
    },
    # During the warm-up time, the flows of the origin-destination pairs defined
    # in od_oscillations will be 0.
    "warm_up_time": 0,
}

DEF_CONFIG = {
    # The user must define the following values:
    # 1. "od_file_path": Path to the origin-destination flow file.
    # 2. "taz_file_path": Path to the taz definitions (mapping taz to edge ids)
    # 3. "network_file_path": Path to the network file.
    #
    # Define periodic flow for certain origin-destination pairs.
    "periodic_flow_config": DEF_PERIODIC_FLOW_CONFIG.copy(),
    # output_dir of None will save all files in the same directory as the
    # sumo_config_file.
    "output_dir": None,
    # Defining output_suffix will add the defined string at the end of the base
    # name of the output files
    "output_suffix": "",
    # Additional files to load to SUMO (such as detectors, calibrators, etc.).
    "additional_files": [],
    # flow_scale is a scaling factor to apply to the values in the input
    # origin-destination flow file (od_file_path).
    "flow_scale": 1,
    # Vehicle depart lane setting (see: https://sumo.dlr.de/docs/Definition_of_Vehicles)
    "depart_lane": "random",
    # Vehicle depart speed setting (see: https://sumo.dlr.de/docs/Definition_of_Vehicles)
    "depart_speed": "desired",
    "depart_pos": "base",
    # Define network segment configurations. The network edges will be divided
    # into segments based on these configurations.
    "segment_config": DEF_SEGMENT_CONFIG.copy(),
    # Turn on single lane mode. This changes the vehicle type parameters such that:
    # 1. Vehicles depart only from the rightmost lane for all origins.
    # 2. Vehicles do not change lanes for cooperation or for increasing their
    #    own speed. Vehicles change lanes only to exit or to merge.
    # 3. Vehicles are more assertive in their lane changes (useful for merges).
    # TODO: Should also divide the flows by the number of lanes for each origin...
    "single_lane_mode": False,
    # Vehicle parameters of human driven vehicles. If None, uses DEF_VEH_PARAMS
    "veh_params": None,
    # Percent of AVs among all vehicles. If 0, all vehicles will be human-driven
    # of type "DEFAULT_VEHTYPE".
    "av_penetration_percentage": 0,
    # AV parameters. If None, will use DEF_VEHICLE_PARAMS with the updates of
    # DEF_AV_PARAMS. Used only if av_penetration_percentage > 0.
    "av_veh_params": None,
    # Switch vehicles from default to AV using a random uniform distribution. If
    # False, switching from default to AV is done uniformly. Used only if
    # av_penetration_percentage > 0.
    "switch_random_veh": True,
    # Random seed to use for switching from default vehicles to AVs. Used only
    # if switch_random_veh is True, and av_penetration_percentage > 0.
    "seed": None,
}

DEF_SIMPLIFIED_CONFIG = {
    "scenario_dir": "scenarios/single_junction",
    "od_flow_file_name": "edge_flows_interval_8400_taz_reduced",
    "network_file_name": "short_merge_lane_separate_exit_lane.net.xml",
    "taz_file_name": "districts.taz.simplified_junctions.xml",
    "output_dir": "scenarios/single_junction/test_calibrated",
    "single_lane": True,
    "av_penetration_percentage": 100,
    "calibrated": False,
    "no_merge": False,
    "keep_veh_names_no_merge": True,
    "default_tau": None,
    "inflow_time_headway": 2,
    "change_lc_av_only": False,
    "no_lc": False,
    "no_lc_right": True,
    "custom_lc": False,
    "lc_config": {
        "lcKeepRight": "0",
        "lcAssertive": "2.5",
        "lcSpeedGain": "5",
        "lcImpatience": "0.7",
    },
    "warm_up_time": 200,
    "merge_flow_duration": 30,
    "merge_flow_percent": 100,
    "break_period_duration": 8400,
    "human_speed_std_0": False,
    "output_suffix": "",
}


def create_sumo_config(config: dict):
    od_file_path = config["od_file_path"]
    network_file_path = config["network_file_path"]
    periodic_flow_config = config.get("periodic_flow_config")
    output_dir = Path(config.get("output_dir", DEF_CONFIG["output_dir"]))

    # Use default vehicle params if not specified
    vehicle_types = config.get("vehicle_types")
    if vehicle_types is None:
        veh_params: dict | None = config.get("veh_params")
        if veh_params is None:
            veh_params = deepcopy(DEF_VEHICLE_PARAMS)
        vehicle_types = {"DEFAULT_VEHTYPE": veh_params}

    output_suffix = config.get("output_suffix", "")
    # Change vehicle params and output_suffix if single lane mode is required.
    single_lane_mode = config.get("single_lane_mode", DEF_CONFIG["single_lane_mode"])
    if single_lane_mode:
        for vtype in vehicle_types.keys():
            vehicle_types[vtype] |= DEF_SINGLE_LANE_VEHICLE_PARAMS
        if not isinstance(config["depart_lane"], dict):
            config["depart_lane"] = "0"
        output_suffix = "_single_lane" + output_suffix

    Path.mkdir(output_dir, exist_ok=True)

    if periodic_flow_config is not None:
        periodic_od, od_file_path = create_and_save_periodical_od(
            od_file_path,
            periodic_flow_config["period_time"],
            periodic_flow_config["od_oscillations"],
            periodic_flow_config["warm_up_time"],
            output_dir=output_dir,
            output_suffix=output_suffix,
        )
    additional_files: set = set(config.get("additional_files", []))

    # Divide into segments and add to additional files
    segment_config: dict = config.get("segment_config")
    if segment_config is not None:
        (
            detector_root,
            detector_file_path,
        ) = create_and_save_detectors_for_highway_profile(
            network_file_path,
            edges=segment_config["edges"],
            segment_nominal_length=segment_config["segment_nominal_length"],
            update_time=segment_config["update_time"],
            detector_results_output_file_name=segment_config[
                "detector_results_output_file_name"
            ],
            use_internal_edges=segment_config.get(
                "use_internal_edges", DEF_SEGMENT_CONFIG["use_internal_edges"]
            ),
            output_dir=output_dir,
        )
        additional_files.add(Path(detector_file_path).name)

    (
        sumo_config_root,
        sumo_config_path,
        route_file_path,
    ) = create_and_save_sumo_config_file(
        od_file_path=od_file_path,
        taz_file_path=config["taz_file_path"],
        network_file_path=network_file_path,
        flow_scale=config.get("flow_scale", DEF_CONFIG["flow_scale"]),
        vehicle_types=vehicle_types,
        output_dir=output_dir,
        additional_files=additional_files,
        depart_lane=config.get("depart_lane", DEF_CONFIG["depart_lane"]),
        depart_speed=config.get("depart_speed", DEF_CONFIG["depart_speed"]),
        depart_pos=config.get("depart_pos", DEF_CONFIG["depart_pos"]),
        output_suffix="",  # output_suffix,
    )

    av_penetration_percentage = config.get("av_penetration_percentage", 0)
    if av_penetration_percentage >= 0:
        av_veh_params = config.get("av_veh_params")
        if av_veh_params is None:
            av_veh_params = deepcopy(DEF_VEHICLE_PARAMS)
            av_veh_params.update(DEF_AV_PARAMS)
            if single_lane_mode:
                av_veh_params.update(DEF_SINGLE_LANE_VEHICLE_PARAMS)

        sumo_config_root, sumo_config_path = add_avs_to_routes(
            sumo_config_path,
            av_veh_params,
            av_penetration_percentage,
            config.get("switch_random_veh", DEF_CONFIG["switch_random_veh"]),
            config.get("seed", DEF_CONFIG["seed"]),
        )

    return sumo_config_path


def get_sumo_config_od_file_stem(simplified_config_overrides):
    simplified_config = DEF_SIMPLIFIED_CONFIG | simplified_config_overrides
    od_file_name = simplified_config["od_flow_file_name"]

    if_single_lane = simplified_config["single_lane"]
    no_merge = simplified_config["no_merge"]
    calibrated = simplified_config["calibrated"]
    keep_veh_names_no_merge = simplified_config["keep_veh_names_no_merge"]
    inflow_time_headway = simplified_config["inflow_time_headway"]

    if if_single_lane:
        od_file_name += "_single_lane"
    if calibrated:
        od_file_name += "_calibrated"
    if no_merge and not keep_veh_names_no_merge:
        od_file_name += "_no_merge"

    od_file_name += f"_{inflow_time_headway}_sec"

    return od_file_name


def get_sumo_config_output_suffix(simplified_config_overrides):
    simplified_config = DEF_SIMPLIFIED_CONFIG | simplified_config_overrides

    merge_flow_percent = simplified_config["merge_flow_percent"]
    if_single_lane = simplified_config["single_lane"]
    no_merge = simplified_config["no_merge"]
    no_lc = simplified_config["no_lc"]
    no_lc_right = simplified_config["no_lc_right"]
    change_lc_av_only = simplified_config["change_lc_av_only"]

    keep_veh_names_no_merge = simplified_config["keep_veh_names_no_merge"]
    default_tau = simplified_config["default_tau"]

    custom_lc = simplified_config["custom_lc"]
    lc_config: dict = simplified_config["lc_config"]
    human_speed_std_0 = simplified_config["human_speed_std_0"]

    output_suffix = simplified_config.get("output_suffix")
    output_suffix = "" if output_suffix is None else output_suffix

    if not merge_flow_percent == 100:
        output_suffix = f"_merge_flow_percent_{merge_flow_percent}" + output_suffix

    if no_merge and keep_veh_names_no_merge:
        output_suffix += "_no_merge"

    if human_speed_std_0:
        output_suffix += "_human_speed_std_0"

    if default_tau is not None:
        output_suffix += f"_tau_{default_tau}"

    if not if_single_lane:
        if no_lc:
            output_suffix += "_no_lc"

        elif no_lc_right:
            output_suffix += "_no_lc_right"

        elif custom_lc:
            output_suffix += (
                "_lc"
                + (
                    f"_right_{lc_config.get('lcKeepRight')}"
                    if lc_config.get("lcKeepRight") is not None
                    else ""
                )
                + (
                    f"_assertive_{lc_config.get('lcAssertive')}"
                    if lc_config.get("lcAssertive") is not None
                    else ""
                )
                + (
                    f"_speed_{lc_config.get('lcSpeedGain')}"
                    if lc_config.get("lcSpeedGain") is not None
                    else ""
                )
                + (
                    f"_impatience_{lc_config.get('lcImpatience')}"
                    if lc_config.get("lcImpatience") is not None
                    else ""
                )
            )

        if change_lc_av_only:
            output_suffix += "_change_lc_av_only"
    return output_suffix


def get_sumo_config_creation_params(simplified_config_overrides: dict):
    simplified_config = DEF_SIMPLIFIED_CONFIG | simplified_config_overrides

    if_single_lane = simplified_config["single_lane"]
    av_penetration_percentage = simplified_config["av_penetration_percentage"]
    no_merge = simplified_config["no_merge"]
    no_lc = simplified_config["no_lc"]
    no_lc_right = simplified_config["no_lc_right"]
    change_lc_av_only = simplified_config["change_lc_av_only"]

    keep_veh_names_no_merge = simplified_config["keep_veh_names_no_merge"]
    default_tau = simplified_config["default_tau"]

    custom_lc = simplified_config["custom_lc"]
    lc_config = simplified_config["lc_config"]
    human_speed_std_0 = simplified_config["human_speed_std_0"]

    working_dir = Path(__file__).parents[1]
    scenario_dir = simplified_config["scenario_dir"]

    od_file_name = get_sumo_config_od_file_stem(simplified_config_overrides) + ".xml"

    # network_file_name = "short_merge_lane.net.xml"
    # network_file_name = "short_merge_lane_exit_connection_fix.net.xml"
    network_file_name = simplified_config["network_file_name"]
    taz_file_name = simplified_config["taz_file_name"]
    output_dir = simplified_config["output_dir"]

    periodic_flow_config_overrides = {
        "period_time": (
            simplified_config["merge_flow_duration"],
            simplified_config["break_period_duration"],
        ),
        "warm_up_time": simplified_config["warm_up_time"],
    }

    config: dict = deepcopy(DEF_CONFIG)

    merge_flow_percent = simplified_config.get("merge_flow_percent")
    if no_merge and keep_veh_names_no_merge:
        merge_flow_percent = 0

    assert merge_flow_percent >= 0, (
        f"merge_flow_percent (requested value: {merge_flow_percent}) must not be negative."
    )

    if merge_flow_percent is not None:
        periodic_flow_config_overrides["od_oscillations"] = {}
        for taz_pair, flow_vals in config["periodic_flow_config"][
            "od_oscillations"
        ].items():
            periodic_flow_config_overrides["od_oscillations"][taz_pair] = {
                flow_key: flow_val * merge_flow_percent / 100
                for flow_key, flow_val in flow_vals.items()
            }

    # We can replace the code below with setting merge_flow_percent=0
    # before the code above when no_merge=True

    # if no_merge and keep_veh_names_no_merge:
    #     periodic_flow_config_overrides |= {
    #         "od_oscillations": {
    #             ("taz_4", "taz_reduced_end"): {"high_scale": 0, "low_scale": 0}
    #         }
    #     }

    od_file_path = working_dir / scenario_dir / od_file_name
    network_file_path = working_dir / scenario_dir / network_file_name
    taz_file_path = working_dir / scenario_dir / taz_file_name

    config["od_file_path"] = od_file_path
    config["taz_file_path"] = taz_file_path
    config["network_file_path"] = network_file_path
    config["additional_files"] = ["throughput_detectors.xml"]
    config["output_dir"] = output_dir
    if no_merge and not keep_veh_names_no_merge:
        config["periodic_flow_config"] = None
    else:
        config["periodic_flow_config"] |= periodic_flow_config_overrides
    config["single_lane_mode"] = if_single_lane
    config["av_penetration_percentage"] = av_penetration_percentage

    config["depart_speed"] = "last"
    if if_single_lane:
        config["depart_lane"] = {"taz_reduced_start_taz_reduced_end": ["1"]}
    else:
        config["depart_lane"] = {
            "taz_reduced_start_taz_reduced_end": ["1", "2", "3", "4"]
        }
    # config["depart_speed"] = "avg"
    # config["depart_speed"] = "desired"
    # config["depart_pos"] = "last"
    config["veh_params"] = DEF_VEHICLE_PARAMS.copy()
    if default_tau is not None:
        config["veh_params"] |= {"tau": str(default_tau)}

    if human_speed_std_0:
        config["veh_params"] |= {"speedDev": "0"}

    if not if_single_lane:
        lc_params = {}
        if no_lc:
            lc_params = DEF_SINGLE_LANE_VEHICLE_PARAMS | {"lcKeepRight": "0"}

        elif no_lc_right:
            lc_params = {"lcKeepRight": "0", "lcAssertive": "3", "lcSpeedGain": "5"}

        elif custom_lc:
            lc_params = lc_config

        config["av_veh_params"] = config["veh_params"] | DEF_AV_PARAMS | lc_params
        if not change_lc_av_only:
            config["veh_params"] |= lc_params

    config["output_suffix"] = get_sumo_config_output_suffix(simplified_config_overrides)
    return config


if __name__ == "__main__":
    simplified_config = {
        "scenario_dir": "scenarios/single_junction",
        "od_flow_file_name": "edge_flows_interval_8400_taz_reduced",
        "network_file_name": "short_merge_lane_separate_exit_lane.net.xml",
        "taz_file_name": "districts.taz.simplified_junctions.xml",
        "output_dir": "scenarios/single_junction/test_calibrated",
        "single_lane": True,
        "av_penetration_percentage": 50,
        "calibrated": False,
        "no_merge": True,
        "keep_veh_names_no_merge": True,
        "default_tau": None,
        "inflow_time_headway": 2,
        "change_lc_av_only": False,
        "no_lc": False,
        "no_lc_right": True,
        "custom_lc": False,
        "lc_config": {
            "lcKeepRight": "0",
            "lcAssertive": "2.5",
            "lcSpeedGain": "5",
            "lcImpatience": "0.7",
        },
        "warm_up_time": 200,
        "merge_flow_duration": 30,
        "break_period_duration": 8400,
    }

    config = get_sumo_config_creation_params(simplified_config)
    test_sumo_cfg_path = create_sumo_config(config)
    print(f"saved to {test_sumo_cfg_path}")
