# %%
from pathlib import Path
import xml.etree.ElementTree as ET
from metrics_utils import save_xml_element

WORKING_DIR = Path(__file__).parents[1]
# working_dir = "Avs_scenario_V2"
# working_dir = "full_sim_new_routes"
# working_dir = "scenarios/reduced_size_simulation_v2"
SCENARIO_DIR = "scenarios/maryam_scenario_full"

NETWORK_FILE = "final_net.net.xml"
# network_file = "new_final_net.net.xml"

if __name__ == "__main__":
    network_path = Path(WORKING_DIR) / SCENARIO_DIR / NETWORK_FILE
    network_root = ET.parse(network_path).getroot()

    additional_file_attrib = {
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/additional_file.xsd",
    }
    detector_root = ET.Element("additional", additional_file_attrib)

    for edge in network_root.iter("edge"):
        if "function" in edge.attrib:
            continue

        for lane in edge.iter("lane"):
            detector_attrib = {
                "id": lane.get("id"),
                "lane": lane.get("id"),
                "pos": "0.5",
                "freq": "60.00",
                "file": "lane_detector_results.xml",
                "vTypes": "DEFAULT_VEHTYPE",
            }
            detector = ET.SubElement(detector_root, "e1Detector", detector_attrib)

    save_xml_element(
        detector_root,
        Path(WORKING_DIR) / SCENARIO_DIR / "add_edge_detectors.xml",
        encoding="UTF-8",
        xml_declaration=True,
    )
