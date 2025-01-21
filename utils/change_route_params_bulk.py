import xml.etree.ElementTree as ET
from pathlib import Path
from metrics_utils import save_xml_element

WORKING_DIR = Path(__file__).parents[1]
SCENARIO_DIR = "scenarios/full_sim_new_routes"
ROUTE_FILE = "background_traffic.rou.xml"

ROUTE_PARAMS = dict(
    departSpeed="desired",
    departLane="free",
)

if __name__ == "__main__":
    route_path = Path(WORKING_DIR) / SCENARIO_DIR / ROUTE_FILE
    route_root = ET.parse(route_path).getroot()

    for vehicle in route_root.iter("vehicle"):
        for attribute, value in ROUTE_PARAMS.items():
            vehicle.set(attribute, value)

save_xml_element(route_root, route_path)
