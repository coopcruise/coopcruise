import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
from utils.metrics_utils import save_xml_element

WORKING_DIR = Path(__file__).parents[1]
SCENARIO_DIR = "scenarios/single_junction"
CONFIG_FILE = (
    "edge_flows_interval_8400_taz_reduced_periodic"
    "_warmup_240_high_120s_low_8400s"
    "_highway_profile_short_merge_lane.sumocfg"
)
# CONFIG_FILE = (
#     "edge_flows_interval_8400_taz_reduced_no_merge"
#     "_highway_profile_short_merge_lane.sumocfg"
# )
SINGLE_LANE_MODE = False
AV_VEH_PARAMS = {
    "id": "AV",
    "speedFactor": "1.25",
    "color": "green",
    "lcSpeedGain": "0" if SINGLE_LANE_MODE else "1",  # "200",
    "lcStrategic": "0" if SINGLE_LANE_MODE else "20.0",  # "1.0",
    "lcAssertive": "20" if SINGLE_LANE_MODE else "5.0",
    "lcCooperative": "-1" if SINGLE_LANE_MODE else "1",
    "carFollowModel": "IDM",
}

AV_PENETRATION_PERCENTAGE = 100
SWITCH_RANDOM_VEH = True


def create_routes_with_avs(
    route_path: Path,
    av_veh_params: dict,
    av_penetration_percentage: int,
    random: bool = True,
    seed: int = None,
):
    if not route_path.exists():
        raise ValueError(f"No file exists at the path {route_path}")

    if av_penetration_percentage < 0 or av_penetration_percentage > 100:
        raise ValueError("av_penetration_percentage must be between 0 and 100")

    required_av_params = ["id"]
    if any([param not in av_veh_params.keys() for param in required_av_params]):
        raise ValueError(
            f"av_veh_params must contain the following parameters: {', '.join(required_av_params)}"
        )

    route_root = ET.parse(route_path).getroot()

    # Add AV vtype to routes
    av_vtype_exists = False
    for vtype in route_root.iter("vType"):
        if vtype.get("id") == av_veh_params["id"]:
            av_vtype_exists = True
            for attribute, value in av_veh_params:
                vtype.set(attribute, value)

    if not av_vtype_exists:
        num_vtypes = len(list(route_root.iter("vType")))
        route_root.insert(num_vtypes, ET.Element("vType", av_veh_params))

    num_vehicles = len(list(route_root.iter("vehicle")))
    num_avs = int(round(num_vehicles * av_penetration_percentage / 100))

    # %% Get indices of AV
    if random:
        rng = np.random.default_rng(seed=seed)
        av_indices = rng.choice(
            np.arange(num_vehicles), size=min(num_avs, num_vehicles), replace=False
        )
    else:
        av_indices = np.arange(
            start=0, stop=num_vehicles, step=int(np.round(av_penetration_percentage))
        )

    # %% Replace corresponding vehicles to the AV type
    for i, vehicle in enumerate(route_root.iter("vehicle")):
        if i in av_indices:
            vehicle.set("type", av_veh_params["id"])

    # %% Save new routes file
    result_name = (
        route_path.name.split(".rou.xml")[0]
        + f"_av_{av_penetration_percentage}_percent.rou.xml"
    )
    save_xml_element(route_root, route_path.parent / result_name)
    return result_name


def add_avs_to_routes(
    sumo_config_path: Path | str,
    av_veh_params: dict,
    av_penetration_percentage: int,
    switch_random_veh: bool,
    seed: int = None,
):
    new_sumo_config_root = ET.parse(sumo_config_path).getroot()
    for routes in new_sumo_config_root.iter("route-files"):
        route_file_names = routes.get("value").split(",")
        new_route_names = []
        route_dir = Path(sumo_config_path).parent
        for route_name in route_file_names:
            route_path = route_dir / route_name
            new_route_names.append(
                create_routes_with_avs(
                    route_path,
                    av_veh_params,
                    av_penetration_percentage,
                    switch_random_veh,
                    seed=seed,
                )
            )
        routes.set("value", ",".join(new_route_names))

    new_config_name = (
        sumo_config_path.name.split(".sumocfg")[0]
        + f"_av_{av_penetration_percentage}_percent.sumocfg"
    )

    new_sumo_config_path = sumo_config_path.parent / new_config_name
    save_xml_element(new_sumo_config_root, new_sumo_config_path)
    return new_sumo_config_root, new_sumo_config_path


if __name__ == "__main__":
    sumo_config_path = Path(WORKING_DIR) / SCENARIO_DIR / CONFIG_FILE
    add_avs_to_routes(
        sumo_config_path, AV_VEH_PARAMS, AV_PENETRATION_PERCENTAGE, SWITCH_RANDOM_VEH
    )
