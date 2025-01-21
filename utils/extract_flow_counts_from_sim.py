import pandas as pd
import numpy as np
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from metrics_utils import save_xml_element

WORKING_DIR = Path(__file__).parents[1]
RESULTS_DIR = "results"
# SCENARIO_NAME = "full_sim_new_routes"
# CONFIG_DIR = "full_new_routes_all_detectors"
# SCENARIO_NAME = "full"
# CONFIG_DIR = ""
SCENARIO_NAME = "maryam_scenario_full"
CONFIG_DIR = "config_file_with_lane_detectors_subscriptions"
FLOW_COUNT_FILE = "lane_detector_flow_counts.csv"
WARM_UP_TIME = 1200  # seconds
FLOW_TIME_INTERVAL = 8400  # seconds
SECONDS_PER_STEP = 0.5  # seconds


def combine_flows(flow_counts: pd.DataFrame, time_steps=None, columns=None):
    time_steps = flow_counts.index if time_steps is None else time_steps
    columns = flow_counts.columns if columns is None else columns

    combined_flows = {}
    for step_flows in flow_counts.loc[time_steps, columns].iterrows():
        valid_flows = step_flows[1][step_flows[1] != "{}"]
        for flow in valid_flows.values:
            for destination, count in json.loads(flow).items():
                if destination in combined_flows.keys():
                    combined_flows[destination] += count
                else:
                    combined_flows[destination] = count
    return combined_flows


if __name__ == "__main__":
    scenario_results_path = Path(WORKING_DIR) / RESULTS_DIR / SCENARIO_NAME / CONFIG_DIR
    flow_counts = pd.read_csv(scenario_results_path / FLOW_COUNT_FILE, index_col=0)

    # %% Create list of edges from lanes, while maintaining order
    edges_with_duplicates = [lane_name[0] for lane_name in flow_counts.columns.str.split("_")]
    edges_with_lane_counts = dict(
        (edge, edges_with_duplicates.count(edge)) for edge in edges_with_duplicates
    )  # dict maintains order of edges (set doesn't)
    edges = list(edges_with_lane_counts.keys())

    # %% Compute flow counts per edge by combining lane flows
    edge_flow_counts = pd.DataFrame(index=flow_counts.index, columns=edges)

    for step_idx, step_flows in enumerate(flow_counts.iterrows()):
        valid_flows = step_flows[1][step_flows[1] != "{}"]
        combined_flows = dict((edge, {}) for edge in edges)
        for column, flow in valid_flows.items():
            edge = column.split("_")[0]
            for destination, count in json.loads(flow).items():
                if destination in combined_flows[edge].keys():
                    combined_flows[edge][destination] += count
                else:
                    combined_flows[edge][destination] = count
        for edge, flows in combined_flows.items():
            edge_flow_counts.iloc[step_idx].loc[edge] = str(flows).replace("'", '"')

    # %% Create OD file
    warm_up_steps = int(WARM_UP_TIME / SECONDS_PER_STEP)
    num_interval_steps = int(FLOW_TIME_INTERVAL / SECONDS_PER_STEP)
    num_steps = len(edge_flow_counts.index)
    num_intervals = max(0, int(np.ceil((num_steps - warm_up_steps) / num_interval_steps)) + 1)

    root_attrib = {
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/datamode_file.xsd",
    }

    flow_root = ET.Element("data", attrib=root_attrib)

    start_step = 0
    for interval_idx in range(num_intervals):
        num_current_interval_steps = warm_up_steps if interval_idx == 0 else num_interval_steps
        end_step = min(start_step + num_current_interval_steps, num_steps)
        interval_attrib = {
            "id": "DEFAULT_VEHICLE",
            "begin": str(start_step * SECONDS_PER_STEP),
            "end": str(end_step * SECONDS_PER_STEP),
        }
        interval = ET.SubElement(flow_root, "interval", interval_attrib)
        flow_chunk = edge_flow_counts.iloc[start_step:end_step]
        for edge in flow_chunk.columns:
            combined_flow = combine_flows(flow_chunk, columns=[edge])
            for destination, count in combined_flow.items():
                flow_attrib = {
                    "from": edge,
                    "to": destination,
                    "count": str(count),
                }
                ET.SubElement(interval, "tazRelation", flow_attrib)

        start_step = end_step

    save_xml_element(
        flow_root,
        scenario_results_path / f"edge_flows_interval_{FLOW_TIME_INTERVAL}.xml",
        encoding="utf-8",
        xml_declaration=True,
    )
