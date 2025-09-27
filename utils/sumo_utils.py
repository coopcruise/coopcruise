import os
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import traci.constants as tc
from typing import List, Dict, Union, Optional


def get_route_file_path(sumo_config_path):
    """Extract SUMO route file path from SUMO configuration file.

    Args:
        sumo_config_path:
            Path to SUMO configuration xml file. Assumes route file path is
            defined within this file under 'input'->'route-files'.

    Returns:
        Path to SUMO route file.
    """
    route_file = list(
        list(ET.parse(sumo_config_path).getroot().iter("input"))[0].iter("route-files")
    )[0].get("value")
    return Path(sumo_config_path).parent / route_file


def get_network_file_path(sumo_config_path):
    """Extract SUMO network file path from SUMO configuration file.

    Args:
        sumo_config_path:
            Path to SUMO configuration xml file. Assumes network file path is
            defined within this file under 'input'->'net-file'.

    Returns:
        Path to SUMO network file.
    """
    network_file = list(
        list(ET.parse(sumo_config_path).getroot().iter("input"))[0].iter("net-file")
    )[0].get("value")
    return Path(sumo_config_path).parent / network_file


def get_edge_num_lanes(network_file_path, no_internal: bool = False):
    """Extract number of lanes for each edge in SUMO network file.

    Args:
        network_file_path:
            Path to network definition xml file.
        no_internal:
            Whether to exclude internal edges from results. Internal edges are
            edges with IDs starting with ':', created automatically for
            junctions between regular edges. Defaults to False.

    Returns:
        A dictionary with edge IDs as keys and number of lanes as values.
    """
    net_tree = ET.parse(network_file_path)
    net_root = net_tree.getroot()
    edge_lanes = {
        edge.get("id"): len(list(edge.iter("lane")))
        for edge in net_root.iter("edge")
        if (not edge.get("id").startswith(":")) or no_internal is False
    }
    return edge_lanes


def get_edge_length(network_file_path, no_internal: bool = False):
    """Extract length of all edges in SUMO network file.

    Args:
        network_file_path:
            Path to network definition xml file.
        no_internal:
            Whether to exclude internal edges from results. Internal edges are
            edges with IDs starting with ':', created automatically for
            junctions between regular edges. Defaults to False.

    Returns:
        A dictionary with edge IDs as keys and edge lengths as values.
    """
    return _get_edge_property(network_file_path, "length", no_internal)


def get_edge_max_speed(network_file_path, no_internal: bool = False):
    """Extract maximum speed of all edges in SUMO network file.

    Args:
        network_file_path:
            Path to network definition xml file.
        no_internal:
            Whether to exclude internal edges from results. Internal edges are
            edges with IDs starting with ':', created automatically for
            junctions between regular edges. Defaults to False.

    Returns:
        A dictionary with edge IDs as keys and edge maximum speed as values.
    """
    return _get_edge_property(network_file_path, "speed", no_internal)


def _get_edge_property(network_file_path, property: str, no_internal: bool = False):
    """Extract a specific property for all edges in SUMO network file.

    Args:
        network_file_path:
            Path to network definition xml file.
        property:
            The name of the property to extract.
        no_internal:
            Whether to exclude internal edges from results. Internal edges are
            edges with IDs starting with ':', created automatically for
            junctions between regular edges. Defaults to False.

    Returns:
        A dictionary with edge IDs as keys and edge property values as values.
    """
    net_tree = ET.parse(network_file_path)
    net_root = net_tree.getroot()
    edge_property = {
        edge.get("id"): float(edge[0].get(property))
        for edge in net_root.iter("edge")
        if not edge.get("id").startswith(":") or no_internal is False
    }
    return edge_property


def get_edge_ids(network_file_path, no_internal: bool = False):
    """Extract edge IDs from SUMO network file.

    Args:
        network_file_path:
            Path to network definition xml file.
        no_internal:
            Whether to exclude internal edges from results. Internal edges are
            edges with IDs starting with ':', created automatically for
            junctions between regular edges. Defaults to False.

    Returns:
        List of edge IDs found in the network file.
    """
    net_tree = ET.parse(network_file_path)
    net_root = net_tree.getroot()
    edge_ids = [
        edge.get("id")
        for edge in net_root.iter("edge")
        if not edge.get("id").startswith(":") or no_internal is False
    ]
    return edge_ids


def get_junction_ids(network_file_path):
    """Extract junction IDs from SUMO network file.

    Args:
        network_file_path:
            Path to network definition xml file.

    Returns:
        A list of junction IDs found in the network file.
    """
    net_tree = ET.parse(network_file_path)
    net_root = net_tree.getroot()
    junction_ids = [junction.get("id") for junction in net_root.iter("junction")]
    return junction_ids


def edge_distance_from_start(network_file_path, ordered_edges):
    """Compute distance of edge terminal points relative to the start of the
    first highway edge.

    Args:
        network_file_path:
            Path to network definition xml file.
        ordered_edges:
            List of edge IDs sorted by their highway position.

    Raises:
        ValueError: If ordered_edges includes an edge not found in the network
        file.

    Returns:
        A tuple of:
            A mapping between edge IDs and location of edge start points as
            values.

            A dictionary with edge IDs as keys and location of edge end points
            as values.

            A dictionary with edge IDs as keys and edge lengths as values.
    """
    edge_lengths = get_edge_length(network_file_path)
    invalid_edges = [edge for edge in ordered_edges if edge not in edge_lengths.keys()]
    if len(invalid_edges) > 0:
        raise ValueError(
            f"The following edges in the route are not in the network: {','.join(invalid_edges)}"
        )
    path_edge_lengths = {edge: edge_lengths[edge] for edge in ordered_edges}
    edge_end_locations = {
        edge: end_location
        for edge, end_location in zip(
            path_edge_lengths.keys(), np.cumsum(list(path_edge_lengths.values()))
        )
    }
    edge_start_locations = {
        edge: edge_end_locations[edge] - path_edge_lengths[edge]
        for edge in path_edge_lengths.keys()
    }
    return edge_start_locations, edge_end_locations, path_edge_lengths


def object_path_location(
    object_data: dict,
    edge_start_location: dict,
    edge_key=tc.VAR_ROAD_ID,
    location_key=tc.VAR_LANEPOSITION,
):
    """Find location of objects measured from the start of a path.

    Args:
        object_data:
            A dictionary with object id as key, and a dictionary of parameters
            as values. The value dictionary must include the key
            `tc.VAR_ROAD_ID` with edge id as a value, and the key
            `tc.VAR_LANEPOSITION` whose value is the location of the object
            along the lane it is on.
        edge_start_location:
            An ordered dictionary with edge ids as keys and start location as
            values. This can be computed using the edge_distance_from_start
            function using the network file and an ordered list of edges in the
            route path.

    Returns:
        A dictionary with object ids as keys whose values are location along the
        road path. If the object is not on any of the path edges defined in
        edge_start_location, its value will be `None`.
    """
    return {
        object_id: (
            float(object_params[location_key])
            + edge_start_location[object_params[edge_key]]
            if object_params[edge_key] in edge_start_location.keys()
            else None
        )
        for object_id, object_params in object_data.items()
    }


def get_detector_ids(detector_file_path):
    """Extract loop detector IDs from a loop detector definition file.

    Args:
        detector_file_path:
            Path to loop detector definition xml file.

    Returns:
        A list of loop detector IDs.
    """
    detectors = ET.parse(detector_file_path).getroot()
    detector_tags = ["e1Detector", "inductionLoop"]
    detector_ids = []
    for tag in detector_tags:
        detector_ids += [detector.get("id") for detector in detectors.iter(tag)]
    return detector_ids


def get_detector_parameters(detector_file_path):
    """Extract loop detector parameters.

    Args:
        detector_file_path:
            Path to loop detector definition xml file.

    Returns:
        A dictionary with detector IDs as keys and detector attributes as
        values.
    """
    detectors = ET.parse(detector_file_path).getroot()
    detector_tags = ["e1Detector", "inductionLoop"]
    detector_params = {}
    for tag in detector_tags:
        detector_params.update(
            {detector.get("id"): detector.attrib for detector in detectors.iter(tag)}
        )

    return detector_params


def get_detector_results_files(sumo_config_path):
    """Extract loop detector result files from SUMO configs.

    Args:
        sumo_config_path:
            Path to SUMO configuration xml file. Assumes loop detector file
            paths are defined within this file under
            'input'->'additional-files'.

    Returns:
        A list of loop detector result file paths used for the scenario.
    """
    config = ET.parse(sumo_config_path).getroot()
    additional_files = []
    for input_params in config.iter("input"):
        for additional in input_params.iter("additional-files"):
            additional_files += [
                file_name.strip() for file_name in additional.get("value").split(",")
            ]

    additional_files = set(additional_files)
    result_files = set()
    config_dir = Path(sumo_config_path).parent
    for file_name in additional_files:
        file_path = config_dir / file_name
        content = ET.parse(file_path).getroot()
        detector_tags = ["e1Detector", "inductionLoop"]
        for tag in detector_tags:
            result_files.update(
                [detector.get("file") for detector in content.iter(tag)]
            )

    return [config_dir / result_file for result_file in result_files]


def detector_group_mapping(
    detector_file_path, sorted_road_edges: List[str] = None
) -> Dict[str, str]:
    """Create a mapping from detector IDs to their edge and position.
    Detectors located at the same position on different lanes will get the same
    value.

    Args:
        detector_file_path:
            Path to detector definition xml file.
        sorted_road_edges (optional):
            List of sorted road edges. If specified, sorts detector mapping by
            edge and location within the edge according to this list. Discards
            all detectors located on edges not in this list. Defaults to None.

    Returns:
        A dictionary with detector IDs as keys, and edges and positions within
        the edges as values in the following format:
            '<edge_id>_pos_<position in meters>'
    """
    detectors = ET.parse(detector_file_path).getroot()
    detector_tags = ["e1Detector", "inductionLoop"]
    detector_map = {}
    for tag in detector_tags:
        detector_map.update(
            {
                detector.get("id"): detector.get("lane").split("_")[0]
                + f"_pos_{detector.get('pos')}"
                for detector in detectors.iter(tag)
            }
        )

    if sorted_road_edges is not None:
        # Sort by edge and position within the edge
        detector_map = {
            key: value
            for key, value in detector_map.items()
            if value.split("_")[0] in sorted_road_edges
        }
        max_position_digits = (
            int(
                np.log10(
                    max(
                        [float(value.split("_")[-1]) for value in detector_map.values()]
                    )
                )
            )
            + 1
        )
        detector_map = dict(
            sorted(
                detector_map.items(),
                key=lambda x: sorted_road_edges.index(x[1].split("_")[0])
                + float(x[1].split("_")[-1]) / (10**max_position_digits),
            )
        )
    return detector_map


def get_detector_section_lengths(
    detector_file_path, network_file_path, sorted_road_edges: List[str]
):
    """Extract mapping between segment IDs to length of segment between each
    detector to the preceding detector.

    Args:
        detector_file_path:
            Path to detector definition xml file.
        network_file_path:
            Path to network definition xml file.
        sorted_road_edges:
            List of sorted road edges of the highway. The road start is assumed
            to be the start of the first edge in this list.

    Returns:
        A dictionary with segment IDs as keys and corresponding segment lengths
        as values. Each segment ID is a string representing the end position of
        the segment in the following format:
            '<edge_id>_pos_<position in meters>'
    """
    detector_map = detector_group_mapping(detector_file_path, sorted_road_edges)
    segment_end_positions_sorted = {
        segment_position_str: {
            "edge": segment_position_str.split("_")[0],
            "position": float(segment_position_str.split("_")[-1]),
        }
        for segment_position_str in detector_map.values()
    }
    edge_start_locations, _, _ = edge_distance_from_start(
        network_file_path, sorted_road_edges
    )
    segment_end_locations = object_path_location(
        segment_end_positions_sorted,
        edge_start_locations,
        edge_key="edge",
        location_key="position",
    )
    segment_len = np.array(list(segment_end_locations.values()))
    segment_len[1:] -= segment_len[:-1]

    return {
        segment_position_str: section_len
        for segment_position_str, section_len in zip(
            segment_end_locations.keys(), segment_len
        )
    }


def get_detector_segment_data(
    detector_file_path, network_file_path, sorted_road_edges: List[str]
):
    """Extract segment data defined by loop detectors along the road.
    Assumes detectors are located along a single road consisting of multiple
    edges.

    Args:
        detector_file_path:
            Path to detector definition xml file.
        network_file_path:
            Path to network definition xml file.
        sorted_road_edges:
            List of sorted road edges of the highway. The road start is assumed
            to be the start of the first edge in this list.

    Returns:
        A dictionary with segment IDs as keys and corresponding segment data as
        values. Each segment ID is a string representing the end position of the
        segment in the following format:
            `"<edge_id>_pos_<position in meters>"`
        The segment data is a dictionary that includes the following keys:
            `"Start"`:
                Segment start point data. The start point data is a dictionary
                with the following keys:
                    `"edge"`:
                        Segment start point edge ID (str).
                    `"edge_position"`:
                        Segment start point position along the edge (float).
                    `"path_position"`:
                        Segment start point position along the entire road path
                        (float).
            `"end"`:
                Segment end point data. The end point data is given in the same
                format as the start point data.
            `"length"`:
                Segment length (float).
    """
    detector_map = detector_group_mapping(detector_file_path, sorted_road_edges)
    segment_end_positions = {
        segment_end_pos: {
            "edge": segment_end_pos.split("_")[0],
            "edge_position": float(segment_end_pos.split("_")[-1]),
        }
        for segment_end_pos in detector_map.values()
    }

    edge_start_locations, _, _ = edge_distance_from_start(
        network_file_path, sorted_road_edges
    )

    segment_end_path_positions = object_path_location(
        segment_end_positions,
        edge_start_locations,
        edge_key="edge",
        location_key="edge_position",
    )

    for segment_id, segment_end_path_position in segment_end_path_positions.items():
        segment_end_positions[segment_id].update(
            {"path_position": segment_end_path_position}
        )

    segment_start_positions = {}
    prev_segment_end = None
    for segment_id, end_position in segment_end_positions.items():
        if prev_segment_end is None:
            segment_start = {
                "edge": end_position["edge"],
                "edge_position": 0,
                "path_position": 0,
            }
        else:
            segment_start = prev_segment_end

        segment_start_positions.update({segment_id: segment_start})
        prev_segment_end = end_position

    return {
        segment_id: {
            "end": segment_end,
            "start": segment_start,
            "length": segment_end["path_position"] - segment_start["path_position"],
        }
        for segment_id, segment_start, segment_end in zip(
            segment_start_positions.keys(),
            segment_start_positions.values(),
            segment_end_positions.values(),
        )
    }


def get_edge_segment_map(segment_data: dict):
    """Create a map between edges and segments within these edges.
    Assumes each segment is defined by its end point edge.

    Args:
        segment_data:
            A dictionary with segment IDs as keys and corresponding segment data
            as values. The value must at least include segment end point data
            under the `"end"` key. The end point data is a dictionary with the
            following keys:
                `"edge"`:
                    Segment end point edge ID (str).
                `"edge_position"`:
                    Segment end point position along the edge.

    Returns:
        A dictionary with edge IDs as keys and segment data as values. Values
        are dictionaries with the following keys:
            `"positions"`:
                A list of segment end positions within the edge.
            `"idx"`:
                A list of segment indices. These indices are global indices of
                all road segments.
    """
    edge_segment_map = {}
    for segment_id, data in segment_data.items():
        segment_end = data["end"]
        segment_end_edge = segment_end["edge"]
        segment_end_edge_position = float(segment_end["edge_position"])
        if segment_end_edge not in edge_segment_map.keys():
            edge_segment_map[segment_end_edge] = {
                "positions": [segment_end_edge_position],
                "segment_ids": [segment_id],
            }
        else:
            edge_segment_map[segment_end_edge]["positions"].append(
                segment_end_edge_position
            )
            edge_segment_map[segment_end_edge]["segment_ids"].append(segment_id)

    num_segments = 0
    for val in edge_segment_map.values():
        val["idx"] = list(range(num_segments, num_segments + len(val["positions"])))
        num_segments += len(val["positions"])

    return edge_segment_map


def extract_num_veh_from_routes(
    route_path: Union[str, Path],
    start_time: float = 0,
    end_time: Optional[float] = None,
):
    """Extract number of departing vehicles from a routes file

    Args:
        route_path:
            Path to the routes file.
        start_time:
            Start time from which to start counting departing vehicles. Defaults
            to 0.
        end_time:
            End time for which to end counting departing vehicles. If None,
            counts up to the end of the file. Defaults to None.

    Returns:
        Number of departing vehicles.
    """
    route_root = ET.parse(route_path)
    num_vehicles_in_range = len(
        [
            vehicle
            for vehicle in route_root.findall("vehicle")
            if float(vehicle.get("depart")) >= start_time
            and (
                float(vehicle.get("depart")) <= end_time
                if end_time is not None
                else True
            )
        ]
    )
    return num_vehicles_in_range


def extract_num_departed_veh_from_routes(
    route_path: Union[str, Path],
    start_time: float = 0,
    end_time: float = None,
    seconds_per_step: float = 0.5,
):
    """Extract array of cumulative departures per step from a routes file

    Args:
        route_path:
            Path to the routes file.
        start_time:
            Start time from which to start counting departing vehicles. Defaults
            to 0.
        end_time:
            End time for which to end counting departing vehicles. If None,
            counts up to the end of the file. Defaults to None.
        seconds_per_step:
            Sumo step length, used for counting departures per step

    Returns:
        Array of cumulative departures per step
    """
    route_root = ET.parse(route_path)
    num_departures = [0] * int((end_time - start_time) // seconds_per_step + 1)

    for vehicle in route_root.findall("vehicle"):
        depart_time = float(vehicle.get("depart"))
        if depart_time >= start_time and (
            depart_time <= end_time if end_time else True
        ):
            num_departures[int(depart_time // seconds_per_step)] += 1

    cum_departures = np.cumsum(num_departures).tolist()

    return cum_departures


def extract_vehicle_departure_time_from_routes(
    route_path: Union[str, Path],
    start_time: float = 0,
    end_time: Optional[float] = None,
):
    """Extract vehicle departure times from a routes file

    Args:
        route_path:
            Path to the routes file.
        start_time:
            Start time from which to start counting departing vehicles. Defaults
            to 0.
        end_time:
            End time for which to end counting departing vehicles. If None,
            counts up to the end of the file. Defaults to None.

    Returns:
        A dictionary with vehicle ids as keys and departure time as values.
    """
    route_root = ET.parse(route_path)
    veh_depart_time = {}
    for vehicle in route_root.findall("vehicle"):
        departure_time = float(vehicle.get("depart"))
        if departure_time >= start_time and (
            departure_time <= end_time if end_time is not None else True
        ):
            veh_depart_time.update({vehicle.get("id"): departure_time})

    return veh_depart_time


def extract_highway_profile_detector_file_name(
    sumo_config_path: Union[str, Path],
    detector_file_name_substr="add_highway_profile_detectors",
):
    """Extract the highway profile detector xml file name from a SUMO config
    file.

    Args:
        sumo_config_path:
            Path to the SUMO config file.
        detector_file_name_substr:
            A substring that must be present in the required file name. Defaults
            to "add_highway_profile_detectors".

    Returns:
        Highway profile detector xml file name
    """
    config_root = ET.parse(sumo_config_path)
    additional_files = [
        s.strip()
        for s in list(config_root.iter("additional-files"))[0].get("value").split(",")
    ]

    for file in additional_files:
        file_name = Path(file).name
        if detector_file_name_substr in file_name:
            return file_name

    return None


def get_veh_type_param(route_path: Path | str, veh_type: str, param_name: str):
    """Extract a parameter value for a vehicle type from a routes file

    Args:
        route_path:
            Path to the routes file.
        veh_type:
            Vehicle type name.
        param_name:
            Parameter name.

    Returns:
        The value of the parameter for the specified vehicle type. Returns None
        if vehicle type or parameter were not found in route file.
    """
    route_root = ET.parse(route_path)
    for vtype in route_root.findall("vType"):
        if not vtype.get("id") == veh_type:
            continue
        if param_name not in vtype.attrib.keys():
            print(
                f"Parameter {param_name} not found for vehicle type {veh_type} in route file {route_path}"
            )
            return None

        return float(vtype.get(param_name))

    print(f"Vehicle type {veh_type} not found in route file {route_path}")
    return None


def extract_vehicle_ids_from_routes(
    route_path: Path | str, starts_with: str = "", ends_with: str = ""
):
    """Extract vehicle ids from routes file

    Args:
        route_path:
            Path to the routes file.
        starts_with (optional):
            Return only vehicle ids that start with this string. Defaults to "".
        ends_with (optional):
            Return only vehicle ids that end with this string. Defaults to "".

    Returns:
        A list of vehicle ids that start with starts_with and end with ends_with
    """
    route_root = ET.parse(route_path)
    vehicle_ids = []
    for vehicle in route_root.findall("vehicle"):
        veh_id = vehicle.get("id")
        if veh_id.startswith(starts_with) and veh_id.endswith(ends_with):
            vehicle_ids.append(veh_id)

    return vehicle_ids


def extract_num_veh_per_type_from_route(route_path: Path | str):
    route_root = ET.parse(route_path)
    veh_type_counts = {
        veh_type: 0 for veh_type in extract_veh_types_from_route(route_path)
    }
    default_veh_type_id = "DEFAULT_VEHTYPE"
    for vehicle in route_root.findall("vehicle"):
        veh_type = (
            vehicle.get("vtype")
            if vehicle.get("vtype") is not None
            else default_veh_type_id
        )
        veh_type_counts[veh_type] += 1

    return veh_type_counts


def extract_veh_types_from_route(route_path: Path | str):
    route_root = ET.parse(route_path)
    vehicle_types = set()
    for veh_type in route_root.findall("vType"):
        veh_type_id = veh_type.get("id")
        vehicle_types.add(veh_type_id)

    return vehicle_types


def get_episode_results_dir(
    results_dir,
    sumo_config_file,
    scenario_dir,
    random_av_switching,
    av_percent,
    name_postfix: str = None,
):
    dir_name = f"{os.path.splitext(os.path.basename(sumo_config_file))[0]}"

    if random_av_switching and av_percent > 0:
        dir_name += f"_random_switch_av_percent_{av_percent}"

    if name_postfix is not None:
        dir_name += f"_{name_postfix}"

    episode_results_dir = Path(results_dir) / Path(scenario_dir).name / dir_name
    return episode_results_dir