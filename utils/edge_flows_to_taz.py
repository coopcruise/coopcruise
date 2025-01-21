# %%
import os
from pathlib import Path
import xml.etree.ElementTree as ET
from i24_utils import get_taz_junction
from metrics_utils import save_xml_element


WORKING_DIR = Path(__file__).parents[1]
SCENARIO_DIR = "scenarios/reduced_sim_exit_59"
TAZ_FILE = "districts.taz.edited.xml"
RESULT_SUFFIX = None  # "_maryam"
# TODO: Compute theses automatically.
TAZ_START_REDUCED_ID = "taz_reduced_start"
TAZ_END_REDUCED_ID = "taz_reduced_end"
JUNCTION_NUM_BEFORE_TAZ_START_REDUCED = 3
JUNCTION_NUM_AFTER_TAZ_END_REDUCED = 6


def edge_to_taz_od_pair(edge_flow_path, edge_to_taz: dict):
    edge_flow_root = ET.parse(edge_flow_path).getroot()

    for interval in edge_flow_root.findall("interval"):
        for flow in interval.findall("tazRelation"):
            origin = flow.get("from")
            destination = flow.get("to")
            if origin not in edge_to_taz.keys():
                interval.remove(flow)
                continue

            if destination not in edge_to_taz.keys():
                interval.remove(flow)
                continue

            if origin == destination:
                interval.remove(flow)
                continue

            flow.set("from", edge_to_taz[origin])
            flow.set("to", edge_to_taz[destination])
    return edge_flow_root


def reduce_taz_routes(
    edge_flow_root: ET.Element,
    junction_num_before_taz_start_reduced=JUNCTION_NUM_BEFORE_TAZ_START_REDUCED,
    junction_num_after_taz_end_reduced=JUNCTION_NUM_AFTER_TAZ_END_REDUCED,
    taz_start_reduced_id=TAZ_START_REDUCED_ID,
    taz_end_reduced_id=TAZ_END_REDUCED_ID,
):
    for interval in edge_flow_root.findall("interval"):
        for od_pair in interval.findall("tazRelation"):
            start_taz = od_pair.get("from")
            end_taz = od_pair.get("to")

            if start_taz == taz_end_reduced_id:
                interval.remove(od_pair)
                continue

            taz_start_junction = (
                junction_num_before_taz_start_reduced + 1
                if start_taz == taz_start_reduced_id
                else get_taz_junction(start_taz)
            )
            taz_end_junction = (
                junction_num_after_taz_end_reduced - 1
                if end_taz == taz_end_reduced_id
                else get_taz_junction(end_taz)
            )

            # remove_eastbound od pairs
            if taz_start_junction > taz_end_junction:
                interval.remove(od_pair)

            elif (
                taz_end_junction <= junction_num_before_taz_start_reduced
                or taz_start_junction >= junction_num_after_taz_end_reduced
            ):
                interval.remove(od_pair)

            elif taz_start_junction <= junction_num_before_taz_start_reduced:
                interval.remove(od_pair)

            if taz_end_junction >= junction_num_after_taz_end_reduced:
                od_pair.set("to", taz_end_reduced_id)

    return edge_flow_root


def combine_repeated_od_pairs(taz_flow_root: ET.Element):
    for interval in taz_flow_root.findall("interval"):
        routes = {}
        for od_pair in interval.findall("tazRelation"):
            start_taz = od_pair.get("from")
            end_taz = od_pair.get("to")
            if (start_taz, end_taz) in routes.keys():
                routes[(start_taz, end_taz)].set(
                    "count",
                    str(int(routes[(start_taz, end_taz)].get("count")) + int(od_pair.get("count"))),
                )
                interval.remove(od_pair)
            else:
                routes[(start_taz, end_taz)] = od_pair

    return taz_flow_root


if __name__ == "__main__":
    taz_path = Path(WORKING_DIR) / SCENARIO_DIR / TAZ_FILE
    taz_root = ET.parse(taz_path).getroot()
    taz_to_edge = {taz.get("id"): taz.get("edges") for taz in taz_root.iter("taz")}
    edge_to_taz = {taz.get("edges"): taz.get("id") for taz in taz_root.iter("taz")}

    # %% Rename origins and destinations to TAZ names and delete unused edges
    edge_flow_path = Path(WORKING_DIR) / EDGE_FLOW_FILE
    taz_flow_root = edge_to_taz_od_pair(edge_flow_path, edge_to_taz)

    # %% reduce and combine taz routes
    taz_flow_root = reduce_taz_routes(taz_flow_root)
    taz_flow_root = combine_repeated_od_pairs(taz_flow_root)

    # %% save results
    edge_flow_file_name = os.path.basename(edge_flow_path)

    new_od_name = (
        os.path.splitext(edge_flow_file_name)[0]
        + "_taz_reduced"
        # + "_westbound"
    )

    if RESULT_SUFFIX is not None:
        new_od_name += RESULT_SUFFIX

    new_od_name += os.path.splitext(edge_flow_file_name)[1]

    save_xml_element(
        taz_flow_root,
        Path(WORKING_DIR) / SCENARIO_DIR / new_od_name,
        encoding="utf-8",
        xml_declaration=True,
    )
