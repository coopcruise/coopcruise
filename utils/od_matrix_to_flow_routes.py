# %% imports
import shutil
import subprocess
import os
from pathlib import Path
import xml.etree.ElementTree as ET
from utils.metrics_utils import save_xml_element

# %% Define default inputs
WORKING_DIR = Path(__file__).parents[1]
# SCENARIO_DIR = "scenarios/reduced_junctions"
SCENARIO_DIR = "scenarios/single_junction"
# NETWORK_FILE = "new_final_net.net.xml"
NETWORK_FILE = "short_merge_lane.net.xml"
TAZ_FILE = "districts.taz.simplified_junctions.xml"
# SCENARIO_DIR = "scenarios/maryam_scenario_full"
# NETWORK_FILE = "final_net.net.xml"
# TAZ_FILE = "districts.taz.edited.xml"
# TAZ_FILE = "districts.taz.simplified_junctions.xml"
# OD_FILE = "edge_flows_interval_8400_taz_reduced_no_exit.xml"
# OD_FILE = "edge_flows_interval_8400_taz_reduced.xml"
OD_FILE = (
    "edge_flows_interval_8400_taz_reduced_periodic_warmup_240_high_240s_low_8400s.xml"
)
# OD_FILE = "edge_flows_interval_8400_taz_reduced_no_merge.xml"
FLOW_SCALE = 1
ADDITIONAL_FILES = [
    "throughput_detectors.xml",
    # "INRIX_add_detectors.xml",
]
ADD_LANE_DETECTORS = False
ACTIVATE_CALIBRATORS = False
ADD_HIGHWAY_PROFILE_DETECTORS = True
IS_SHORT_MERGE_LANE = True
SINGLE_LANE_MODE = False
DEPART_LANE = "0" if SINGLE_LANE_MODE else "random"  # "free"
DEPART_SPEED = "desired"  # "max"
DEPART_POS = "base"

# %% Define vehicle types
VEHICLE_TYPES = {
    "DEFAULT_VEHTYPE": {
        "id": "DEFAULT_VEHTYPE",
        "color": "grey",
        "carFollowModel": "IDM",
        "lcSpeedGain": "0" if SINGLE_LANE_MODE else "1",  # "200",
        "lcStrategic": "0" if SINGLE_LANE_MODE else "20.0",  # "1.0",
        "lcAssertive": "20" if SINGLE_LANE_MODE else "5.0",
        "lcCooperative": "-1" if SINGLE_LANE_MODE else "1",
        "insertionChecks": "collision",
        "speedFactor": "1.25",
    },
    "AV": {
        "id": "AV",
        "color": "grey",
        "carFollowModel": "IDM",
        "lcSpeedGain": "0" if SINGLE_LANE_MODE else "1",  # "200",
        "lcStrategic": "0" if SINGLE_LANE_MODE else "20.0",  # "1.0",
        "lcAssertive": "20" if SINGLE_LANE_MODE else "5.0",
        "lcCooperative": "-1" if SINGLE_LANE_MODE else "1",
        "lcKeepRight": "0",
        "insertionChecks": "collision",
        "speedFactor": "1.25",
    },
}


def create_partial_flow_routes(
    od_root: ET.Element,
    taz_to_edge: dict,
    vehicle_types: dict = VEHICLE_TYPES,
    flow_scale=FLOW_SCALE,
    depart_lane=DEPART_LANE,
    depart_speed=DEPART_SPEED,
    depart_pos=DEPART_POS,
) -> ET.Element:
    routes_root = ET.Element("routes")

    for attr in vehicle_types.values():
        ET.SubElement(routes_root, "vType", attr)

    flow_count = 0
    for time_interval in od_root.iter("interval"):
        interval_attributes = {
            "begin": time_interval.get("begin"),
            "end": time_interval.get("end"),
        }
        veh_type = time_interval.get("id")
        interval = ET.SubElement(routes_root, "interval", interval_attributes)

        for od_pair in time_interval.iter("tazRelation"):
            number = int(int(od_pair.get("count")) * flow_scale)
            start_taz = od_pair.get("from")
            end_taz = od_pair.get("to")
            if isinstance(depart_lane, dict):
                from_to = start_taz + "_" + end_taz
                from_to_depart_lane = depart_lane.get(from_to, ["free"])
                if not isinstance(from_to_depart_lane, list):
                    from_to_depart_lane = [from_to_depart_lane]
                for lane in from_to_depart_lane:
                    flow_attributes = {
                        "id": f"{veh_type}_{flow_count}",
                        "fromTaz": start_taz,
                        "toTaz": end_taz,
                        "number": str(int(number / len(from_to_depart_lane))),
                        "departLane": lane,
                        "departSpeed": depart_speed,
                        "departPos": depart_pos,
                    }
                    flow = ET.SubElement(interval, "flow", flow_attributes)
                    ET.SubElement(
                        flow,
                        "route",
                        {"edges": f"{taz_to_edge[start_taz]} {taz_to_edge[end_taz]}"},
                    )
                    flow_count += 1

            else:
                flow_attributes = {
                    "id": f"{veh_type}_{flow_count}",
                    "fromTaz": start_taz,
                    "toTaz": end_taz,
                    "number": str(number),
                    "departLane": depart_lane,
                    "departSpeed": depart_speed,
                    "departPos": depart_pos,
                }
                flow = ET.SubElement(interval, "flow", flow_attributes)
                ET.SubElement(
                    flow,
                    "route",
                    {"edges": f"{taz_to_edge[start_taz]} {taz_to_edge[end_taz]}"},
                )
                flow_count += 1
    return routes_root


def create_sumo_config(
    route_file,
    network_file=NETWORK_FILE,
    additional_files=None,
    start_time=0,
    end_time=15000,
    random_depart_offset=0,
    extrapolate_depart_pos=True,
    max_num_vehicles=-1,
    time_to_teleport=-1,
    default_car_following_model="IDM",
    show_detectors=True,
):
    sumo_cfg_root = ET.Element("configuration")

    cfg_input = ET.SubElement(sumo_cfg_root, "input")
    ET.SubElement(cfg_input, "net-file", dict(value=network_file))
    ET.SubElement(cfg_input, "route-files", dict(value=route_file))
    if additional_files is not None and len(additional_files) > 0:
        ET.SubElement(
            cfg_input, "additional-files", dict(value=",".join(additional_files))
        )

    cfg_time = ET.SubElement(sumo_cfg_root, "time")
    ET.SubElement(cfg_time, "begin", dict(value=str(start_time)))
    ET.SubElement(cfg_time, "end", dict(value=str(end_time)))

    cfg_process = ET.SubElement(sumo_cfg_root, "processing")
    ET.SubElement(
        cfg_process, "random-depart-offset", dict(value=str(random_depart_offset))
    )
    ET.SubElement(
        cfg_process, "extrapolate-departpos", dict(value=str(extrapolate_depart_pos))
    )
    ET.SubElement(cfg_process, "max-num-vehicles", dict(value=str(max_num_vehicles)))
    ET.SubElement(cfg_process, "time-to-teleport", dict(value=str(time_to_teleport)))
    ET.SubElement(
        cfg_process, "default.carfollowmodel", dict(value=default_car_following_model)
    )

    ET.SubElement(
        sumo_cfg_root, "tls.actuated.show-detectors", dict(value=str(show_detectors))
    )

    return sumo_cfg_root


def create_and_save_sumo_config_file(
    od_file_path: Path | str,
    taz_file_path: Path | str,
    network_file_path: Path | str,
    flow_scale: float = 1,
    vehicle_types: dict[str, dict[str, str]] = None,
    output_dir: Path | str = None,
    additional_files: list[str] = None,
    depart_lane: str = DEPART_LANE,
    depart_speed: str = DEPART_SPEED,
    depart_pos: str = DEPART_POS,
    output_suffix: str = None,
):
    # load OD and TAZ files
    od_root = ET.parse(od_file_path).getroot()
    taz_root = ET.parse(taz_file_path).getroot()

    # Compute mapping from TAZ to edges.
    taz_to_edge = {taz.get("id"): taz.get("edges") for taz in taz_root.iter("taz")}
    vehicle_types = VEHICLE_TYPES if vehicle_types is None else vehicle_types

    # Create route file with flows and partial routes (only start and end edges)
    routes_root = create_partial_flow_routes(
        od_root,
        taz_to_edge,
        vehicle_types,
        flow_scale,
        depart_lane,
        depart_speed,
        depart_pos,
    )

    od_file_name = Path(od_file_path).name
    output_dir = Path(od_file_path).parent if output_dir is None else Path(output_dir)
    input_file = os.path.basename(od_file_name)
    output_file_name = os.path.splitext(input_file)[0]
    if flow_scale != 1:
        output_file_name += f"_flow_scale_{flow_scale}"

    output_suffix = "" if output_suffix is None else output_suffix

    flow_routes_file_name = "initial_route_flow" + output_suffix
    flow_routes_file = f"{output_file_name}_{flow_routes_file_name}.rou.xml"
    flow_route_path = output_dir / flow_routes_file

    save_xml_element(
        routes_root, flow_route_path, encoding="utf-8", xml_declaration=True
    )

    route_file_name = f"{output_file_name + output_suffix}.rou.xml"
    route_file_path = output_dir / route_file_name

    cmd = [
        "duarouter",
        "--repair",
        f"--route-files={flow_route_path}",
        f"--net={network_file_path}",
        f"--taz-files={taz_file_path}",
        f"--output-file={route_file_path}",
        # "--write-trips"
    ]
    subprocess.run(cmd)

    # create SUMO configuration file
    additional_files = [] if additional_files is None else additional_files
    network_file_name = Path(network_file_path).name

    if not (output_dir / network_file_name).exists():
        shutil.copy(network_file_path, output_dir / network_file_name)

    # Copy additional files to output directory if they don't already exist there, and
    # if they exist in the origin-destination directory.
    if additional_files is not None and len(additional_files) > 0:
        for file in additional_files:
            file_name = Path(file).name
            if not (output_dir / file_name).exists():
                search_directories = Path(od_file_path).parents
                for directory in search_directories:
                    file_path = Path(directory) / file_name
                    if file_path.exists():
                        shutil.copy(file_path, output_dir / file_name)
                        break

    sumo_cfg_root = create_sumo_config(
        route_file_name, network_file_name, additional_files
    )
    sumo_cfg_path = output_dir / f"{output_file_name + output_suffix}.sumocfg"
    save_xml_element(
        sumo_cfg_root, sumo_cfg_path, encoding="utf-8", xml_declaration=True
    )

    return sumo_cfg_root, sumo_cfg_path, route_file_path


if __name__ == "__main__":
    od_file_path = Path(WORKING_DIR) / SCENARIO_DIR / OD_FILE
    taz_file_path = Path(WORKING_DIR) / SCENARIO_DIR / TAZ_FILE
    network_file_path = Path(WORKING_DIR) / SCENARIO_DIR / NETWORK_FILE

    if ADD_LANE_DETECTORS:
        ADDITIONAL_FILES.append("add_edge_detectors.xml")

    if ACTIVATE_CALIBRATORS:
        ADDITIONAL_FILES.append("additional__calibrator_file_1_14_2021.xml")

    output_prefix = ""
    if ADD_HIGHWAY_PROFILE_DETECTORS:
        output_prefix += "_highway_profile"
        if IS_SHORT_MERGE_LANE:
            ADDITIONAL_FILES.append(
                "short_merge_lane_add_highway_profile_detectors.xml"
            )
            output_prefix += "_short_merge_lane"
        else:
            ADDITIONAL_FILES.append("add_highway_profile_detectors.xml")

    if SINGLE_LANE_MODE:
        output_prefix += "_single_lane"

    create_and_save_sumo_config_file(
        od_file_path,
        taz_file_path,
        network_file_path,
        FLOW_SCALE,
        VEHICLE_TYPES,
        additional_files=ADDITIONAL_FILES,
        depart_lane=DEPART_LANE,
        depart_speed=DEPART_SPEED,
        output_suffix=output_prefix,
    )
