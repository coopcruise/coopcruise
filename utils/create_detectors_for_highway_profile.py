# %%
import shutil
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
from utils.metrics_utils import save_xml_element
from utils.i24_utils import get_main_road_west_edges
from utils.sumo_utils import get_edge_length

WORKING_DIR = Path(__file__).parents[1]
# SCENARIO_DIR = "scenarios/maryam_scenario_full"
# NETWORK_FILE = "final_net.net.xml"
SCENARIO_DIR = "scenarios/reduced_junctions"
NETWORK_FILE = "new_final_net.net.xml"

DEF_DETECTOR_RESULTS_OUTPUT_FILE_NAME = "speed_profile_detector_results.xml"
DEF_SEGMENT_NOMINAL_LENGTH = 100  # m
DEF_UPDATE_TIME = 60  # seconds
DEF_USE_INTERNAL_EDGES = False
I24_HIGHWAY_WESTBOUND_EDGES = get_main_road_west_edges()


def compute_num_segments_per_edge(
    edge_lengths: np.ndarray | list, segment_length: float = 100
):
    edge_lengths = (
        np.array(edge_lengths) if isinstance(edge_lengths, list) else edge_lengths
    )
    min_num_segments = np.maximum(1, (edge_lengths // segment_length))
    max_num_segments = np.maximum(1, (edge_lengths // segment_length) + 1)

    err_1 = np.abs(edge_lengths / min_num_segments - segment_length)
    err_2 = np.abs(edge_lengths / max_num_segments - segment_length)

    num_segments = np.ones_like(min_num_segments)
    num_segments[err_1 <= err_2] = min_num_segments[err_1 <= err_2]
    num_segments[err_1 > err_2] = max_num_segments[err_1 > err_2]

    return num_segments


def floor_decimals(a, decimals=0):
    return np.true_divide(np.floor(a * 10**decimals), 10**decimals)


def create_detectors_for_highway_profile(
    network_path: Path | str,
    edges: list[str],
    segment_nominal_length: float = DEF_SEGMENT_NOMINAL_LENGTH,
    update_time: int = DEF_UPDATE_TIME,
    detector_results_output_file_name: str = DEF_DETECTOR_RESULTS_OUTPUT_FILE_NAME,
    use_internal_edges: bool = DEF_USE_INTERNAL_EDGES,
):
    network_root = ET.parse(network_path).getroot()
    network_file = Path(network_path).name
    additional_file_attrib = {
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/additional_file.xsd",
    }
    detector_root = ET.Element("additional", additional_file_attrib)

    if not use_internal_edges:
        edges = [edge for edge in edges if not edge.startswith(":")]

    edge_lengths = get_edge_length(network_path, no_internal=not use_internal_edges)
    highway_edge_lengths = {edge: edge_lengths[edge] for edge in edges}
    highway_edge_segments = {
        edge_id: num_segments
        for edge_id, num_segments in zip(
            edges,
            compute_num_segments_per_edge(
                list(highway_edge_lengths.values()), segment_nominal_length
            ),
        )
    }

    for edge in network_root.iter("edge"):
        if "function" in edge.attrib and not use_internal_edges:
            continue

        if edge.get("id") not in edges:
            continue

        detector_positions = floor_decimals(
            highway_edge_lengths[edge.get("id")]
            / highway_edge_segments[edge.get("id")]
            * (np.arange(highway_edge_segments[edge.get("id")]) + 1),
            1,
        )

        for i, position in enumerate(detector_positions):
            for lane in edge.iter("lane"):
                detector_attrib = {
                    "id": f'{lane.get("id")}_{i}',
                    "lane": lane.get("id"),
                    "pos": str(position),
                    "freq": str(update_time),
                    "file": network_file.split(".")[0]
                    + "_"
                    + detector_results_output_file_name,
                    # "vTypes": "DEFAULT_VEHTYPE",
                }
                ET.SubElement(detector_root, "e1Detector", detector_attrib)

    return detector_root


def save_detectors_for_highway_profile(
    detector_root: ET.Element, network_path, output_dir=None
):
    output_dir = Path(network_path).parent if output_dir is None else Path(output_dir)
    network_file = Path(network_path).name
    if not (output_dir / network_file).exists():
        shutil.copy(network_path, output_dir / network_file)

    save_path = (
        output_dir / f"{network_file.split('.')[0]}_add_highway_profile_detectors.xml"
    )
    save_xml_element(detector_root, save_path, encoding="UTF-8", xml_declaration=True)

    return save_path


def create_and_save_detectors_for_highway_profile(
    network_path: Path | str,
    edges: list[str],
    segment_nominal_length: float,
    update_time: int,
    detector_results_output_file_name: str,
    use_internal_edges: bool = False,
    output_dir: Path | str = None,
):
    detector_root = create_detectors_for_highway_profile(
        network_path,
        edges,
        segment_nominal_length,
        update_time,
        detector_results_output_file_name,
        use_internal_edges,
    )

    save_path = save_detectors_for_highway_profile(
        detector_root, network_path, output_dir
    )

    return detector_root, save_path


if __name__ == "__main__":
    create_and_save_detectors_for_highway_profile(
        Path(WORKING_DIR) / SCENARIO_DIR / NETWORK_FILE,
        I24_HIGHWAY_WESTBOUND_EDGES,
        DEF_SEGMENT_NOMINAL_LENGTH,
        DEF_UPDATE_TIME,
        DEF_DETECTOR_RESULTS_OUTPUT_FILE_NAME,
        DEF_USE_INTERNAL_EDGES,
    )
