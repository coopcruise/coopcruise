import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from types import NoneType

from utils.sim_utils import get_sumo_config_file_name
from utils.sumo_utils import (
    get_edge_ids,
    detector_group_mapping,
    get_detector_section_lengths,
    # get_detector_segment_data,
    # get_edge_num_lanes,
    # get_edge_length,
)
from utils.i24_utils import get_main_road_west_edges
from utils.metrics_utils import (
    compute_throughput,
    plot_time_space_diagrams,
    plot_throughput,
)

FIG_SAVE_FORMATS = ["svg", "pdf", "png"]


def get_sim_results_dir_nested(
    common_sim_params: dict,
    compare_within_sim_params: dict | NoneType = None,
    compare_between_sim_params: dict | NoneType = None,
):
    if (
        compare_between_sim_params is not None
        and len(compare_between_sim_params.keys()) > 0
    ):
        if len(compare_between_sim_params.keys()) > 1:
            result_dirs = {}
            param_name, param_vals = list(compare_between_sim_params.items())[0]
            compare_between_sim_params_other = {
                name: vals
                for name, vals in compare_between_sim_params.items()
                if not name == param_name
            }
            for param_val in param_vals:
                nested_result_dirs = get_sim_results_dir_nested(
                    common_sim_params | {param_name: param_val},
                    compare_within_sim_params,
                    compare_between_sim_params_other,
                )

                for name, val in nested_result_dirs.items():
                    if isinstance(name[0], str):
                        new_name = (
                            tuple([param_name] + [name[0]]),
                            tuple([param_val] + [name[1]]),
                        )
                    else:
                        new_name = (
                            tuple([param_name] + list(name[0])),
                            tuple([param_val] + list(name[1])),
                        )
                    result_dirs |= {new_name: val}

            return result_dirs

        return {
            (param_name, param_val): get_sim_results_dir_nested(
                common_sim_params | {param_name: param_val},
                compare_within_sim_params,
                {
                    name: vals
                    for name, vals in compare_between_sim_params.items()
                    if not name == param_name
                },
            )
            for param_name, param_vals in compare_between_sim_params.items()
            for param_val in param_vals
        }

    if (
        compare_within_sim_params is not None
        and len(compare_within_sim_params.keys()) > 0
    ):
        result_dirs = {}
        param_name, param_vals = list(compare_within_sim_params.items())[0]
        compare_within_sim_params_other = {
            name: vals
            for name, vals in compare_within_sim_params.items()
            if not name == param_name
        }
        for param_val in param_vals:
            if len(compare_within_sim_params_other.keys()) > 0 or valid_params(
                common_sim_params | {param_name: param_val}
            ):
                nested_result_dirs = get_sim_results_dir_nested(
                    common_sim_params | {param_name: param_val},
                    compare_within_sim_params_other,
                    compare_between_sim_params,
                )
                if isinstance(nested_result_dirs, dict):
                    for name, val in nested_result_dirs.items():
                        if isinstance(name[0], str):
                            new_name = (
                                tuple([param_name] + [name[0]]),
                                tuple([param_val] + [name[1]]),
                            )
                        else:
                            new_name = (
                                tuple([param_name] + list(name[0])),
                                tuple([param_val] + list(name[1])),
                            )
                        result_dirs |= {new_name: val}
                else:
                    result_dirs |= {(param_name, param_val): nested_result_dirs}
        return result_dirs

    return get_sim_results_dir(common_sim_params)


def get_sim_results_dir(sim_params: dict):
    sumo_config_file_name, sumo_config_file_path = get_sumo_config_file_name(sim_params)
    custom_name_postfix = sim_params.get("custom_name_postfix")
    use_learned_control = (
        sim_params.get("use_learned_control")
        if sim_params.get("use_learned_control") is not None
        else False
    )
    single_lane = (
        sim_params.get("single_lane")
        if sim_params.get("single_lane") is not None
        else False
    )
    use_vsl_control = (
        sim_params.get("use_vsl_control")
        if sim_params.get("use_vsl_control") is not None
        else False
    )
    use_tau_control = (
        sim_params.get("use_tau_control")
        if sim_params.get("use_tau_control") is not None
        else False
    )
    tau_control_only_rightmost_lane = (
        sim_params.get("tau_control_only_rightmost_lane")
        if sim_params.get("tau_control_only_rightmost_lane") is not None
        else False
    )
    tau_val = sim_params.get("tau_val")
    name_postfix = ""
    if custom_name_postfix is not None:
        name_postfix = custom_name_postfix
    elif not any([use_learned_control, use_vsl_control, use_tau_control]):
        name_postfix = "no_control"
    elif use_learned_control:
        name_postfix = "rl_control"
    elif use_vsl_control:
        name_postfix = "vsl_control"
    elif use_tau_control:
        name_postfix = f"tau_control_{tau_val}"
        if tau_control_only_rightmost_lane and not single_lane:
            name_postfix += "_rightmost"

    random_av_switching = sim_params["random_av_switching"]
    random_av_switching_seed = sim_params["random_av_switching_seed"]

    if random_av_switching and random_av_switching_seed is not None:
        name_postfix += f"_av_switch_seed_{random_av_switching_seed}"

    results_dir_name = sumo_config_file_name.split(".sumocfg")[0]
    av_percent = sim_params["av_percent"]
    if random_av_switching and av_percent > 0:
        results_dir_name += f"_random_switch_av_percent_{av_percent}"

    results_dir_name += f"_{name_postfix}"

    results_dir_path = (
        Path("results") / sumo_config_file_path.parent.name / results_dir_name
    )
    return str(results_dir_path)


def valid_params(params: dict):
    no_merge = params.get("no_merge") if params.get("no_merge") is not None else False
    use_learned_control = (
        params.get("use_learned_control")
        if params.get("use_learned_control") is not None
        else False
    )
    use_tau_control = (
        params.get("use_tau_control")
        if params.get("use_tau_control") is not None
        else False
    )
    tau_val = params.get("tau_val")
    use_vsl_control = (
        params.get("use_vsl_control")
        if params.get("use_vsl_control") is not None
        else False
    )
    control_types = [use_learned_control, use_tau_control, use_vsl_control]
    if len([control_type for control_type in control_types if control_type]) > 1:
        return False
    if no_merge and use_learned_control:
        return False
    if no_merge and use_tau_control:
        return False
    if not use_tau_control and tau_val is not None:
        return False
    if use_tau_control and tau_val is None:
        return False

    return True


def extract_sim_group_metadata(sims: dict):
    metadata_file = "metadata.json"
    metadata = {
        sim_name: json.load(open(Path(results_dir) / metadata_file))
        for sim_name, results_dir in sims.items()
        if (Path(results_dir) / metadata_file).exists()
    }
    return metadata


def extract_sim_group_results_summary(sims: dict):
    result_summary_file = "episode_result.json"

    result_summary = {
        sim_name: json.load(open(Path(results_dir) / result_summary_file))
        for sim_name, results_dir in sims.items()
        if (Path(results_dir) / result_summary_file).exists()
    }
    return result_summary


def extract_steps_per_second(metadata: dict):
    steps_per_second = {
        sim_name: int(1 / sim_meta["sumo_config"]["seconds_per_step"])
        for sim_name, sim_meta in metadata.items()
    }
    return steps_per_second


def read_sim_data_file(sim_results_dir, data_file, steps_per_second):
    if (Path(sim_results_dir) / data_file).exists():
        sim_data = pd.read_csv(Path(sim_results_dir) / data_file, index_col=0)
        time_axis = (
            sim_data.index / steps_per_second
            if sim_data.index[-1] != (len(sim_data.index) - 1) / steps_per_second
            else sim_data.index
        )
        return sim_data.set_axis(time_axis)
    return None


def extract_sim_logs(
    sim_group_dirs: dict,
    sim_group_steps_per_second: dict,
    extract_only_veh_travel_info=False,
):
    edge_veh_counts = {}
    edge_avg_speed = {}
    detector_counts = {}
    computation_time = {}
    profile_veh_counts = {}

    segment_avg_speed = {}
    segment_num_veh = {}
    segment_density = {}

    segment_speed_limit = {}
    veh_travel_info = {}

    edge_veh_count_file = "segment_counter.csv"
    avg_speed_file = "average_speed.csv"
    detector_counts_file = "loop_detector_veh_count.csv"
    computation_time_file = "step_computation_time.csv"
    speed_profile_veh_count_file = "speed_profile_detector_step_count.csv"

    segment_avg_speed_file = "segment_avg_speed.csv"
    segment_num_veh_file = "segment_num_veh.csv"
    segment_density_file = "segment_density.csv"

    segment_speed_limit_file = "segment_speed_limit.csv"
    veh_travel_info_file = "veh_travel_info.csv"

    for sim_name in sim_group_dirs.keys():
        results_dir = sim_group_dirs[sim_name]
        steps_per_second = sim_group_steps_per_second[sim_name]

        if not extract_only_veh_travel_info:
            edge_veh_counts[sim_name] = read_sim_data_file(
                results_dir, edge_veh_count_file, steps_per_second
            )

            edge_avg_speed[sim_name] = read_sim_data_file(
                results_dir, avg_speed_file, steps_per_second
            )

            detector_counts[sim_name] = read_sim_data_file(
                results_dir, detector_counts_file, steps_per_second
            )

            computation_time_data = read_sim_data_file(
                results_dir, computation_time_file, steps_per_second
            )
            if computation_time_data is None:
                raise ValueError(f"No data for {results_dir}")

            computation_time[sim_name] = computation_time_data.iloc[:, 0]

            profile_veh_counts[sim_name] = read_sim_data_file(
                results_dir, speed_profile_veh_count_file, steps_per_second
            )

            segment_avg_speed[sim_name] = read_sim_data_file(
                results_dir, segment_avg_speed_file, steps_per_second
            )

            segment_num_veh[sim_name] = read_sim_data_file(
                results_dir, segment_num_veh_file, steps_per_second
            )

            segment_density[sim_name] = read_sim_data_file(
                results_dir, segment_density_file, steps_per_second
            )

            segment_speed_limit[sim_name] = read_sim_data_file(
                results_dir, segment_speed_limit_file, steps_per_second
            )

        if (Path(results_dir) / veh_travel_info_file).exists():
            veh_travel_info[sim_name] = pd.read_csv(
                Path(results_dir) / veh_travel_info_file, index_col=0
            )

    return {
        "edge_veh_counts": edge_veh_counts,
        "edge_avg_speed": edge_avg_speed,
        "detector_counts": edge_avg_speed,
        "computation_time": computation_time,
        "profile_veh_counts": profile_veh_counts,
        "segment_avg_speed": segment_avg_speed,
        "segment_num_veh": segment_num_veh,
        "segment_density": segment_density,
        "segment_speed_limit": segment_speed_limit,
        "veh_travel_info": veh_travel_info,
    }


def compute_sim_group_throughput(
    sim_logs: dict,
    detector_group_map: dict,
    speed_profile_detector_file_path: dict,
    sims_steps_per_second: dict,
    warm_up_time=0,
    throughput_computation_period_time=30,
):
    segment_throughput = {
        sim_name: compute_throughput(
            sim_counts,
            throughput_computation_period_time,
            seconds_per_step=1 / sims_steps_per_second[sim_name],
            warm_up_time=warm_up_time,
            detector_map=detector_group_map[sim_name],
        )[0]
        for sim_name, sim_counts in zip(
            sim_logs["profile_veh_counts"].keys(),
            sim_logs["profile_veh_counts"].values(),
        )
        if sim_name in speed_profile_detector_file_path.keys()
    }
    return segment_throughput


def plot_inflow_outflow(
    segment_throughput: dict,
    ordered_available_segments,
    warm_up_time=0,
    throughput_computation_period_time=30,
):
    inflow_segment = ordered_available_segments[2]
    outflow_segment = ordered_available_segments[-2]

    colormap = mpl.colormaps["tab10"]
    if len(segment_throughput.keys()) > 10:
        colormap = mpl.colormaps["tab20"]

    max_throughput = max(
        [sim_throughput.values.max() for sim_throughput in segment_throughput.values()]
    )

    ax_inflow = None
    for i, (sim_name, sim_throughput) in enumerate(segment_throughput.items()):
        fig_inflow, ax_inflow = plot_throughput(
            sim_throughput[inflow_segment],
            throughput_computation_period_time=throughput_computation_period_time,
            warm_up_time=warm_up_time,
            color=colormap(i % colormap.N),
            label=sim_name,
            y_min=0,
            y_max=max_throughput * 1.05,
            ax=ax_inflow,
            title="Inflow",
        )

    ax_inflow.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    ax_outflow = None
    for i, (sim_name, sim_throughput) in enumerate(segment_throughput.items()):
        fig_outflow, ax_outflow = plot_throughput(
            sim_throughput[outflow_segment],
            throughput_computation_period_time=throughput_computation_period_time,
            warm_up_time=warm_up_time,
            color=colormap(i % colormap.N),
            label=sim_name,
            y_min=0,
            y_max=max_throughput * 1.05,
            ax=ax_outflow,
            title="Outflow",
        )

    ax_outflow.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    return fig_inflow, fig_outflow


def analyze_sim_group(
    sim_group_dirs: dict,
    save_dir: str | Path = None,
    sim_group_name: str = None,
    warm_up_time=0,
    throughput_computation_period_time=30,
):
    metadata = extract_sim_group_metadata(sim_group_dirs)
    # result_summary = extract_sim_group_results_summary(sim_group_dirs)
    sims_steps_per_second = extract_steps_per_second(metadata)
    sim_logs = extract_sim_logs(sim_group_dirs, sims_steps_per_second)

    network_path = {
        sim_name: sim_metadata["network_file"]
        for sim_name, sim_metadata in metadata.items()
    }

    # edge_length = {
    #     sim_name: get_edge_length(sim_network_path, no_internal=False)
    #     for sim_name, sim_network_path in network_path.items()
    # }

    speed_profile_detector_file_path = {
        sim_name: sim_metadata["speed_profile_detector_file_path"]
        for sim_name, sim_metadata in metadata.items()
        if "speed_profile_detector_file_path" in sim_metadata.keys()
    }

    network_road_edges = {
        sim_name: get_edge_ids(sim_network_path)
        for sim_name, sim_network_path in network_path.items()
    }
    main_road_west_segments = get_main_road_west_edges(with_internal=False)

    main_road_west_segments = {
        sim_name: [
            segment
            for segment in main_road_west_segments
            if segment in list(sim_network_road_edges)
        ]
        for sim_name, sim_network_road_edges in network_road_edges.items()
        if sim_name in speed_profile_detector_file_path.keys()
    }

    detector_group_map = {
        sim_name: detector_group_mapping(
            sim_speed_profile_detector_file_path,
            sorted_road_edges=main_road_west_segments[sim_name],
        )
        for sim_name, sim_speed_profile_detector_file_path in speed_profile_detector_file_path.items()
        if sim_name in speed_profile_detector_file_path.keys()
    }

    # segment_data = {
    #     sim_name: get_detector_segment_data(
    #         sim_speed_profile_detector_file_path,
    #         network_path[sim_name],
    #         main_road_west_segments[sim_name],
    #     )
    #     for sim_name, sim_speed_profile_detector_file_path in speed_profile_detector_file_path.items()
    # }

    # edge_num_lanes = {
    #     sim_name: get_edge_num_lanes(sim_network_path)
    #     for sim_name, sim_network_path in network_path.items()
    # }

    # segment_num_lanes = {
    #     sim_name: {
    #         segment_name: sim_edge_num_lanes[segment_val["end"]["edge"]]
    #         for segment_name, segment_val in segment_data[sim_name].items()
    #     }
    #     for sim_name, sim_edge_num_lanes in edge_num_lanes.items()
    # }

    detector_section_len = {
        sim_name: get_detector_section_lengths(
            sim_logs["speed_profile_detector_file_path"][sim_name],
            sim_network_path,
            main_road_west_segments[sim_name],
        )
        for sim_name, sim_network_path in network_path.items()
        if sim_name in speed_profile_detector_file_path.keys()
    }

    ordered_available_segments = [
        segment
        for segment in list(detector_section_len.values())[0].keys()
        if segment in list(sim_logs["segment_density"].values())[0].columns
    ]

    # available_segment_len = {
    #     sim_name: {
    #         segment: segment_length[segment] for segment in ordered_available_segments
    #     }
    #     for sim_name, segment_length in detector_section_len.items()
    # }

    segment_throughput = compute_sim_group_throughput(
        sim_logs,
        detector_group_map,
        speed_profile_detector_file_path,
        sims_steps_per_second,
        warm_up_time,
        throughput_computation_period_time,
    )

    fig_throughput_diagram = plot_time_space_diagrams(
        # data=segment_throughput,
        data={
            sim_name: sim_throughput_data.loc[:, ordered_available_segments[:-1]]
            for sim_name, sim_throughput_data in segment_throughput.items()
        },
        sections=list(detector_section_len.values())[0].keys(),
        sims_steps_per_second=sims_steps_per_second,
        section_lengths=detector_section_len,
        data_name="throughput [veh/hour]",
        title=f"Throughput ({throughput_computation_period_time}s window)",
        max_subplot_title_len=40,
        # max_val=1800,
        # num_rows=1,
    )

    Path(save_dir).mkdir(exist_ok=True)
    for format in FIG_SAVE_FORMATS:
        fig_throughput_diagram.savefig(
            Path(save_dir) / f"{sim_group_name}_throughput_diagrams.{format}"
        )

    fig_inflow, fig_outflow = plot_inflow_outflow(
        segment_throughput,
        ordered_available_segments,
        warm_up_time,
        throughput_computation_period_time,
    )

    for format in FIG_SAVE_FORMATS:
        fig_inflow.savefig(Path(save_dir) / f"{sim_group_name}_inflow.{format}")
        fig_outflow.savefig(Path(save_dir) / f"{sim_group_name}_outflow.{format}")

    fig_segment_avg_speed = plot_time_space_diagrams(
        # data=segment_avg_speed,
        data={
            sim_name: sim_average_speed.loc[:, ordered_available_segments[:-1]]
            for sim_name, sim_average_speed in sim_logs["segment_avg_speed"].items()
        },
        sections=list(detector_section_len.values())[0].keys(),
        sims_steps_per_second=sims_steps_per_second,
        section_lengths=detector_section_len,
        data_name="Average speed [m/s]",
        min_val=0,
        title="Average speed",
        max_subplot_title_len=25,
        max_color_bar_title_len=15,
        # num_rows=1,
    )

    for format in FIG_SAVE_FORMATS:
        fig_segment_avg_speed.savefig(
            Path(save_dir) / f"{sim_group_name}_avg_speed_diagrams.{format}"
        )


def plot_mixed_stat_results(
    mixed_stats: dict,
    title: str = None,
    std_scale_factor=1,
    ax: plt.Axes | None = None,
    label: str | None = None,
):
    if ax is None:
        fig, ax = plt.subplots()
        ax: plt.Axes
        add_title = True
        line_color = "tab:blue"
        error_color = "tab:cyan"
    else:
        fig = ax.get_figure()
        add_title = False
        line_color = "tab:purple"
        error_color = "tab:pink"
    # ax_single_mixed_stat_xlabel = list(multi_lane_mixed_stats_rightmost.keys())[0].split("=")[0].replace("_"," ")
    ax_mixed_stat_xlabel = "ACC-equipped vehicle percentage [%]"
    ax_mixed_stat_x_vals = [
        int(key.split("=")[1]) if isinstance(key, str) else key
        for key in mixed_stats.keys()
    ]
    ax_mixed_stat_y_means = [value["mean"] for value in mixed_stats.values()]
    ax_mixed_stat_y_std = [value["std"] for value in mixed_stats.values()]

    plot_kwargs = dict(marker="*", linewidth=2, markersize=10, color=line_color)

    yerr = np.array(ax_mixed_stat_y_std) * std_scale_factor

    plot_kwargs |= dict(yerr=yerr, ecolor=error_color, capsize=5)  # "lightblue")

    if label is not None:
        plot_kwargs.update(dict(label=label))

    ax.errorbar(ax_mixed_stat_x_vals, ax_mixed_stat_y_means, **plot_kwargs)
    ax.axhline(
        0,
        linestyle="--",
        color="black",
        zorder=-1,
    )

    if add_title:
        if title is not None:
            ax.set_title(
                title,  # mixed autonomy performance",
                fontsize="xx-large",
                fontweight="bold",
            )
        ax.set_xlabel(ax_mixed_stat_xlabel, fontsize="x-large")
        ax.set_ylabel("Average velocity relative change [%]", fontsize="x-large")

    return fig, ax


def compute_sim_travel_metrics(sim_logs):
    veh_travel_data_df = {}
    time_delay_per_vehicle = {}
    time_delay_per_second_per_vehicle = {}
    avg_vel_per_vehicle = {}
    avg_vel_reduction_per_vehicle = {}
    valid_veh = {}
    avg_vel_reduction_per_vehicle_ref = {}
    avg_time_delay = {}
    avg_time_delay_per_second = {}
    avg_vel = {}
    avg_vel_reduction = {}
    avg_vel_reduction_ref = {}

    for sim_group_name, sim_group_logs in sim_logs.items():
        veh_travel_info = sim_group_logs["veh_travel_info"]
        veh_travel_data_df[sim_group_name] = pd.DataFrame(
            {
                (info_name, sim_name): values
                for sim_name, travel_info in veh_travel_info.items()
                for info_name, values in travel_info.to_dict().items()
            }
        )
        # (
        #     veh_travel_data_df["total_time"].iloc[:, 1:]
        #     - veh_travel_data_df["total_time"].iloc[:, 0].values
        # )
        time_delay_per_vehicle[sim_group_name] = (
            veh_travel_data_df[sim_group_name]["total_time"].iloc[:, 1:].T
            - veh_travel_data_df[sim_group_name]["total_time"].iloc[:, 0]
        ).T

        time_delay_per_second_per_vehicle[sim_group_name] = (
            time_delay_per_vehicle[sim_group_name]
            / veh_travel_data_df[sim_group_name]["total_time"].iloc[:, 1:]
        )
        avg_vel_per_vehicle[sim_group_name] = (
            veh_travel_data_df[sim_group_name]["travel_distance"]
            / veh_travel_data_df[sim_group_name]["total_time"]
        )
        avg_vel_reduction_per_vehicle[sim_group_name] = (
            (
                avg_vel_per_vehicle[sim_group_name].iloc[:, 1:].T
                - avg_vel_per_vehicle[sim_group_name].iloc[:, 0]
            )
            / avg_vel_per_vehicle[sim_group_name].iloc[:, 0]
        ).T

        valid_veh[sim_group_name] = (
            veh_travel_data_df[sim_group_name]["travel_distance"].iloc[:, 1] > 100
        )
        avg_vel_reduction_per_vehicle_ref[sim_group_name] = (
            (
                (
                    avg_vel_per_vehicle[sim_group_name].iloc[:, 1:].T
                    - avg_vel_per_vehicle[sim_group_name].iloc[:, 1]
                ).T
            )
            .where(valid_veh[sim_group_name])
            .T
            / avg_vel_per_vehicle[sim_group_name]
            .iloc[:, 1]
            .where(valid_veh[sim_group_name])
        ).T

        avg_time_delay[sim_group_name] = time_delay_per_vehicle[sim_group_name].mean(
            axis=0
        )
        avg_time_delay_per_second[sim_group_name] = time_delay_per_second_per_vehicle[
            sim_group_name
        ].mean(axis=0)
        avg_vel[sim_group_name] = avg_vel_per_vehicle[sim_group_name].mean(axis=0)
        avg_vel_reduction[sim_group_name] = avg_vel_reduction_per_vehicle[
            sim_group_name
        ].mean(axis=0)
        avg_vel_reduction_ref[sim_group_name] = avg_vel_reduction_per_vehicle_ref[
            sim_group_name
        ].mean(axis=0)

    return {
        "avg_time_delay": avg_time_delay,
        "avg_time_delay_per_second": avg_time_delay_per_second,
        "avg_vel": avg_vel,
        "avg_vel_reduction": avg_vel_reduction,
        "avg_vel_reduction_ref": avg_vel_reduction_ref,
    }


def analyze_multiple_sim_groups(
    multi_sim_group_dirs: dict,
    save_all_sim_group_results=False,
    save_dir: str | Path = None,
    multi_sim_group_name: str = "",
):
    metadata = {
        sim_group_name: extract_sim_group_metadata(sim_group_dirs)
        for sim_group_name, sim_group_dirs in multi_sim_group_dirs.items()
    }
    multi_sim_group_steps_per_second = {
        sim_group_name: extract_steps_per_second(sim_group_metadata)
        for sim_group_name, sim_group_metadata in metadata.items()
    }
    sim_logs = {
        sim_group_name: extract_sim_logs(
            sim_group_dirs,
            multi_sim_group_steps_per_second[sim_group_name],
            extract_only_veh_travel_info=save_all_sim_group_results,
        )
        for sim_group_name, sim_group_dirs in multi_sim_group_dirs.items()
    }

    sim_travel_metrics = compute_sim_travel_metrics(sim_logs)
    avg_vel_reduction_ref = sim_travel_metrics["avg_vel_reduction_ref"]

    compare_between_sims_param_names = list(avg_vel_reduction_ref.keys())[0][0]
    compare_switch_seeds = False
    if "random_av_switching_seed" in compare_between_sims_param_names:
        compare_switch_seeds = True

    compare_av_percent = False
    if "av_percent" in compare_between_sims_param_names:
        av_percent_param_idx = compare_between_sims_param_names.index("av_percent")
        compare_av_percent = True

    if compare_av_percent and compare_switch_seeds:
        label = None
        performance_df: pd.DataFrame = (
            (pd.DataFrame(avg_vel_reduction_ref) * 100).T.droplevel(0).T
        )

        av_percent_available = sorted(
            set([key[av_percent_param_idx] for key in performance_df.columns])
        )
        performance = {
            av_percent: {"mean": 0, "std": 0} for av_percent in av_percent_available
        }

        for av_percent in performance.keys():
            av_percent_perf = performance_df[
                [
                    key
                    for key in performance_df.columns
                    if key[av_percent_param_idx] == av_percent
                ]
            ]
            best_strategy = av_percent_perf.T.mean().iloc[1:].idxmax()
            performance[av_percent]["mean"] = av_percent_perf.T.mean()[best_strategy]
            performance[av_percent]["std"] = av_percent_perf.T.std()[
                best_strategy
            ] / np.sqrt(len(av_percent_perf.columns))

            fig_mixed_stat, ax_mixed_stat = plot_mixed_stat_results(
                performance, multi_sim_group_name, 1.96, label=label
            )
            ax_mixed_stat.set_xlim(left=0 - 100 * 0.05, right=100 * 1.05)

        Path(save_dir).mkdir(exist_ok=True)
        for format in FIG_SAVE_FORMATS:
            fig_mixed_stat.savefig(
                Path(save_dir)
                / f"{multi_sim_group_name.lower().replace(' ', '_')}_performance.{format}"
            )
