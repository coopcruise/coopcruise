import xml.etree.ElementTree as ET
from pathlib import Path
from copy import deepcopy
from utils.metrics_utils import save_xml_element

# %% Define default inputs
WORKING_DIR = Path(__file__).parents[1]
# SCENARIO_DIR = "scenarios/reduced_junctions"
SCENARIO_DIR = "scenarios/single_junction"
OD_FILE = "edge_flows_interval_8400_taz_reduced.xml"
PERIOD_TIME = (240, 8400)  # sec
OD_OSCILLATIONS = {("taz_4", "taz_reduced_end"): {"high_scale": 1.2, "low_scale": 0}}
WARM_UP_TIME = 240


def create_periodical_od(
    od_root: ET.Element,
    period_time: int | tuple,
    od_oscillations: dict[tuple, dict[str, float]],
    warm_up_time: int = 0,
):
    # Assume the OD is for the entire period (only one interval).
    periodic_od_root = deepcopy(od_root)
    intervals = periodic_od_root.findall("interval")
    for interval in intervals:
        periodic_od_root.remove(interval)

    # Break into intervals defined by a time period.
    parsed_time = 0
    num_periods = 0
    if isinstance(period_time, int):
        period_time_high = period_time
        period_time_low = period_time
    else:
        period_time_high = period_time[0]
        period_time_low = period_time[1]

    warm_up_done = True
    if warm_up_time > 0:
        period_remainder = warm_up_time
        warm_up_done = False
    else:
        period_remainder = period_time_high
    for time_interval in od_root.iter("interval"):
        interval_start = int(time_interval.get("begin"))
        interval_end = int(time_interval.get("end"))
        while interval_end > parsed_time:
            # Create new interval
            period_duration = min(period_remainder, interval_end - parsed_time)
            new_interval = deepcopy(time_interval)
            new_interval.set("begin", str(parsed_time))
            new_interval.set("end", str(parsed_time + period_duration))
            for od_pair in new_interval.iter("tazRelation"):
                # Scale O/D pair count by the relative interval duration
                count = int(od_pair.get("count"))
                rel_duration = period_duration / (interval_end - interval_start)
                multiplier = 1
                # Scale specific O/D pairs by the high or low scales defined in
                # od_oscillations
                from_to = (od_pair.get("from"), od_pair.get("to"))
                if from_to in od_oscillations.keys():
                    if not warm_up_done:
                        multiplier = 0
                    else:
                        multiplier = (
                            od_oscillations[from_to]["high_scale"]
                            if num_periods % 2 == 0
                            else od_oscillations[from_to]["low_scale"]
                        )

                od_pair.set("count", str(int(rel_duration * count * multiplier)))
            periodic_od_root.append(new_interval)
            parsed_time += period_duration
            # If entire period was completed, add to period counter.
            period_remainder -= period_duration
            if period_remainder == 0:
                if not warm_up_done:
                    warm_up_done = True
                else:
                    num_periods += 1
                period_remainder = (
                    period_time_high if num_periods % 2 == 0 else period_time_low
                )

    return periodic_od_root


def get_periodic_file_name(
    od_file_path: Path | str,
    period_time: int | tuple,
    warm_up_time: int = 0,
    output_dir: Path | str = None,
    output_suffix: str = None,
):
    if isinstance(period_time, int):
        period_time_high = period_time
        period_time_low = period_time
    else:
        period_time_high = period_time[0]
        period_time_low = period_time[1]

    od_file = Path(od_file_path).name
    output_dir = Path(od_file_path).parent if output_dir is None else Path(output_dir)

    output_suffix = "" if output_suffix is None else output_suffix
    if not output_suffix.startswith("_"):
        output_suffix = f"_{output_suffix}"

    periodic_od_path = output_dir / (
        od_file.split(".xml")[0]
        + f"_periodic_warmup_{warm_up_time}_high_{period_time_high}s_low_{period_time_low}s{output_suffix}.xml"
    )
    return periodic_od_path


def create_and_save_periodical_od(
    od_file_path: Path | str,
    period_time: int | tuple,
    od_oscillations: dict[tuple, dict[str, float]],
    warm_up_time: int = 0,
    output_dir: Path | str = None,
    output_suffix: str = None,
):
    # %% load OD and TAZ files
    od_root = ET.parse(od_file_path).getroot()

    periodic_od_path = get_periodic_file_name(
        od_file_path, period_time, warm_up_time, output_dir, output_suffix
    )
    periodic_od = create_periodical_od(
        od_root, period_time, od_oscillations, warm_up_time
    )
    save_xml_element(
        periodic_od, periodic_od_path, encoding="utf-8", xml_declaration=True
    )
    return periodic_od, periodic_od_path


if __name__ == "__main__":
    # %% load OD and TAZ files
    create_and_save_periodical_od(
        Path(WORKING_DIR) / SCENARIO_DIR / OD_FILE,
        PERIOD_TIME,
        OD_OSCILLATIONS,
        WARM_UP_TIME,
    )
