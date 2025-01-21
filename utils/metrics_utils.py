import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from pathlib import Path
import xml.etree.ElementTree as ET
from textwrap import fill
from typing import Dict, Iterable


SECONDS_PER_HOUR = 3600
DEFAULT_SECONDS_PER_STEP = 1


def compute_average_speed(
    segment_counter: pd.DataFrame, average_speed: pd.DataFrame, warm_up_time=0
):
    num_vehicles_total = segment_counter.sum(axis=1)
    avg_speed_total = pd.Series(
        np.zeros((len(segment_counter.index),)), index=segment_counter.index
    )
    valid_time_steps = segment_counter.index[num_vehicles_total != 0]
    avg_speed_values = np.average(
        a=average_speed.loc[valid_time_steps].values,
        axis=1,
        weights=segment_counter.loc[valid_time_steps].values,
    )
    avg_speed_total.loc[valid_time_steps] = avg_speed_values

    simulation_avg_speed = 0
    after_warm_up_valid_time_steps = valid_time_steps[valid_time_steps >= warm_up_time]
    if len(after_warm_up_valid_time_steps > 0):
        simulation_avg_speed = np.average(
            avg_speed_total[after_warm_up_valid_time_steps].values,
            weights=num_vehicles_total[after_warm_up_valid_time_steps].values,
        )

    return avg_speed_total, simulation_avg_speed


def step_count_from_interval_count(
    current_interval_counts: pd.DataFrame, interval_time=60
):
    final_interval_step_idx = current_interval_counts.index % interval_time == (
        interval_time - current_interval_counts.index[1]
    )
    if len(final_interval_step_idx) == 0:
        # Simulation ends before the first interval time is reached
        last_step_interval_counts = 0
    else:
        last_step_interval_counts = current_interval_counts.loc[final_interval_step_idx]
        cumulative_count_steps = (
            last_step_interval_counts.cumsum()
            .reindex(current_interval_counts.index)
            .ffill()
            .fillna(0)
        )

    cumulative_count = current_interval_counts + cumulative_count_steps
    cumulative_count.loc[final_interval_step_idx] -= last_step_interval_counts
    vehicle_count_from_interval = cumulative_count.copy(deep=True)
    vehicle_count_from_interval.iloc[1:] -= vehicle_count_from_interval.iloc[:-1].values
    return vehicle_count_from_interval


def compute_throughput(
    loop_detector_veh_count: pd.DataFrame,
    throughput_computation_period_time=300,
    seconds_per_step=DEFAULT_SECONDS_PER_STEP,
    warm_up_time=0,
    detector_map=None,
):
    if detector_map is None:
        loop_detector_counts_per_step = loop_detector_veh_count.sum(axis=1)
    else:
        loop_detector_counts_per_step = (
            loop_detector_veh_count.T.groupby(detector_map).sum().T
        )

    cumulative_detector_counts = loop_detector_counts_per_step.cumsum()
    cumulative_detector_counts_window = cumulative_detector_counts.copy(deep=True)

    num_throughput_computation_period_steps = int(
        throughput_computation_period_time / seconds_per_step
    )
    if num_throughput_computation_period_steps < len(cumulative_detector_counts):
        cumulative_detector_counts_window.iloc[
            num_throughput_computation_period_steps:
        ] = (
            cumulative_detector_counts.iloc[num_throughput_computation_period_steps:]
            - cumulative_detector_counts.iloc[
                :-num_throughput_computation_period_steps
            ].values
        )

    throughput_window = (
        cumulative_detector_counts_window
        / num_throughput_computation_period_steps
        / seconds_per_step
        * SECONDS_PER_HOUR
    )
    throughput_window.iloc[:num_throughput_computation_period_steps] = (
        cumulative_detector_counts_window.iloc[:num_throughput_computation_period_steps]
        / (
            np.arange(
                np.min(
                    [num_throughput_computation_period_steps, len(throughput_window)]
                )
            )
            + 1
        ).reshape(
            [-1] + [1] * (len(cumulative_detector_counts_window.shape) - 1)
        )  # expand to 2d if there is more than one column
        / seconds_per_step
        * SECONDS_PER_HOUR
    )

    simulation_average_throughput = throughput_window[
        warm_up_time * seconds_per_step :
    ].mean()

    return throughput_window, simulation_average_throughput


def metric_timeseries_plot(
    data: pd.Series,
    warm_up_time: int = None,
    color=None,
    ylabel: str = None,
    label: str = None,
    y_min: float = None,
    y_max: float = None,
    title: str = None,
    ax: plt.Axes = None,
):
    if ax is None:
        warm_up_time = 0 if warm_up_time is None else warm_up_time
        warm_up_color = "tab:grey"
        fig, ax = plt.subplots()
        ax: plt.Axes
        data.plot(color=color, label=label, ax=ax)
        if warm_up_time > 0:
            ax.axvline(warm_up_time, linestyle=":", color=warm_up_color)
            # y_min, y_max = ax.get_ylim()
            y_min = ax.get_ylim()[0] if y_min is None else y_min
            y_max = ax.get_ylim()[1] if y_max is None else y_max
            ax.fill_between(
                [0, warm_up_time], y_min, y_max, alpha=0.1, color=warm_up_color
            )
            ax.text(
                warm_up_time / 2,
                y_min + (y_max - y_min) * 1,
                "Warm up",
                color=warm_up_color,
                fontweight="bold",
                fontsize="x-large",
                horizontalalignment="center",
                verticalalignment="bottom",
            )
            ax.set_ylim(bottom=y_min, top=y_max)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize="x-large")

        ax.set_xlabel("Time steps [s]", fontsize="x-large")
        ax.set_xlim(left=max(0, data.index[0] - 1), right=data.index[-1])

        y_bot = min(ax.get_ylim()[0], y_min) if y_min is not None else ax.get_ylim()[0]
        y_top = max(ax.get_ylim()[1], y_max) if y_max is not None else ax.get_ylim()[1]
        ax.set_ylim(bottom=y_bot, top=y_top)
        plt.xticks(fontsize="x-large")
        plt.yticks(fontsize="x-large")
        default_ratio = fig.get_figwidth() / fig.get_figheight()
        if warm_up_time > 0:
            fig.set_figwidth(
                max(
                    fig.get_figwidth()
                    * 0.6
                    * 1200
                    / warm_up_time
                    * (ax.get_xlim()[1] - ax.get_xlim()[0])
                    / 3000,
                    fig.get_figheight() * default_ratio,
                )
            )

        if title is not None:
            ax.set_title(
                title,
                loc="center",
                y=1.05,
                fontsize="xx-large",
                fontweight="bold",
            )

        fig.tight_layout()
    else:
        fig = ax.get_figure()
        data.plot(color=color, label=label, ax=ax)

    return fig, ax


def plot_mean_speed(
    avg_speed_total,
    warm_up_time=0,
    simulation_avg_speed=None,
    color=None,
    label=None,
    ax: plt.Axes = None,
    title=None,
):
    color = "tab:blue" if color is None else color
    if ax is None:
        fig, ax = metric_timeseries_plot(
            avg_speed_total,
            warm_up_time=warm_up_time,
            color=color,
            ylabel="Average speed [m/s]",
            label=label,
            title=title,
        )
    else:
        fig = ax.get_figure()
        avg_speed_total.plot(
            color=color,
            label=label,
            ax=ax,
        )

    if simulation_avg_speed is not None:
        ax.axhline(simulation_avg_speed, color=color, linestyle="--")

    return fig, ax


def plot_throughput(
    throughput_window: pd.Series,
    throughput_computation_period_time=60,
    warm_up_time=0,
    simulation_average_throughput=None,
    color=None,
    label=None,
    y_min=None,
    y_max=None,
    ax: plt.Axes = None,
    title=None,
):
    color = "tab:blue" if color is None else color
    if ax is None:
        fig, ax = metric_timeseries_plot(
            throughput_window,
            warm_up_time=warm_up_time,
            color="tab:blue",
            ylabel="Throughput [vehicles/hour]",
            label=label,
            y_min=y_min,
            y_max=y_max,
            title=title,
        )

        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        ax.text(
            x_min + (x_max - x_min) * 0.5,
            y_min + (y_max - y_min) * 1,
            f"Throughput measured during a {throughput_computation_period_time} step period",
            color="tab:blue",
            fontweight="bold",
            fontsize="x-large",
            horizontalalignment="center",
            verticalalignment="bottom",
        )
    else:
        fig = ax.get_figure()
        throughput_window.plot(
            color=color,
            label=label,
            ax=ax,
        )

    if simulation_average_throughput is not None:
        ax.axhline(simulation_average_throughput, color="tab:blue", linestyle="--")

    return fig, ax


def plot_throughput_and_speed(
    throughput_window,
    throughput_computation_period_time,
    avg_speed_total,
    warm_up_time=0,
    simulation_average_throughput=None,
    simulation_avg_speed=None,
    throughput_color=None,
    speed_color=None,
):
    throughput_color = "tab:blue" if throughput_color is None else throughput_color
    speed_color = "tab:orange" if speed_color is None else speed_color

    fig, ax_throughput = plot_throughput(
        throughput_window,
        throughput_computation_period_time,
        warm_up_time,
        simulation_average_throughput,
        color=throughput_color,
    )

    ax_speed = ax_throughput.twinx()
    _, ax_speed = plot_mean_speed(
        avg_speed_total,
        warm_up_time,
        simulation_avg_speed,
        color=speed_color,
        ax=ax_speed,
    )

    plt.xticks(fontsize="x-large")
    plt.yticks(fontsize="x-large")

    ax_throughput.set_ylabel(ax_throughput.get_ylabel(), color=throughput_color)
    ax_throughput.tick_params("y", colors=throughput_color)
    ax_speed.set_ylabel("Average speed [m/s]", color=speed_color, fontsize="x-large")
    ax_speed.tick_params("y", colors=speed_color)

    fig.tight_layout()

    return fig, ax_throughput, ax_speed


def multi_savefig(fig: plt.Figure, fig_name, results_dir, formats, **kwargs):
    Path(results_dir).mkdir(exist_ok=True, parents=True)
    if isinstance(formats, str):
        formats = [formats]

    for format in formats:
        fig.savefig(Path(results_dir) / f"{fig_name}.{format}", **kwargs)


def indent(elem: ET.Element, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def save_xml_element(root: ET.Element, save_path, **kwargs):
    indent(root)
    ET.ElementTree(root).write(save_path, **kwargs)


def plot_time_space_diagrams(
    data: Dict[str, pd.DataFrame],
    sections: Iterable[str],
    sims_steps_per_second: Dict[str, int],
    section_lengths: Dict[str, Dict[str, float]],
    data_name: str = None,
    min_val: float = None,
    max_val: float = None,
    length_resolution: int = 50,
    cmap="RdYlGn",
    max_subplot_title_len: int = 35,
    max_color_bar_title_len: int = 10,
    title: str = None,
    data_scale=1,
    num_rows: int = None,
    write_all_axis_labels=False,
    axis_label_size=12,
    sim_title_size=16,
    tick_label_size=12,
):
    """Plot a time-space diagram with time [seconds] in the x axis and location [km] in the y axis.

    Args:
        data:
            A dictionary of data to plot.
            The keys are simulation names, and values are data of each simulation.
            Each simulation data is a pandas DataFrame with time values in seconds as index
            and section names as columns.
        sections:
            A list of sections (ordered) for plotting.
        sims_steps_per_second:
            A dictionary with simulation name as keys and the number of steps per second as values.
        section_lengths:
            A dictionary with simulation name as keys and a dictionary of section length in meters as values.
        data_name (optional):
            The name for the data which will be used as a title for the color bar.
            If None, uses
        min_val (optional):
            Minimum value of data for color normalization.
            If None, uses the minimum value seen in the data.
            Defaults to None.
        max_val (optional):
            Maximum value of data for color normalization.
            If None, uses the maximum value seen in the data.
            Defaults to None.
        length_resolution (optional):
            Resolution for the y axis in meters. Must be an integer.
            Lower is more precise but produces larger images with slower rendering.
            Defaults to 50.
        max_subplot_title_len (optional):
            Maximum number of letters in a single row of a sub-blot title.
            Defaults to 30.
        max_color_bar_title_len (optional):
            Maximum number of letters in a single row of the color bar title.
            Defaults to 10.
        title (optional):
            Title of the cluster of diagrams.
        data_scale (optional):
            Scaling factor for the data. Defaults to 1.


    Returns:
        The matplotlib figure of the space-time diagram.
    """
    data = {
        sim_name: sim_data
        for sim_name, sim_data in data.items()
        if sim_data is not None
    }
    min_val = (
        min(
            [
                sim_data.filter(regex=("^[^:]")).values.min() * data_scale
                for sim_data in data.values()
            ]
        )
        if min_val is None
        else min_val
    )
    max_val = (
        max(
            [
                sim_data.filter(regex=("^[^:]")).values.max() * data_scale
                for sim_data in data.values()
            ]
        )
        if max_val is None
        else max_val
    )

    num_plots = len(data)
    num_rows = (
        int(np.round(np.sqrt(num_plots)))
        if num_rows is None
        else min(num_plots, num_rows)
    )
    num_cols = int(np.ceil(num_plots / num_rows))
    fig, axes = plt.subplots(num_rows, num_cols)

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    fig.subplots_adjust(right=0.83)
    for j, (ax, sim_name, sim_data) in enumerate(
        zip(axes.flatten(), data.keys(), data.values())
    ):
        steps_per_second = sims_steps_per_second[sim_name]
        ax: plt.Axes
        sections_to_plot = [
            section for section in sections if section in sim_data.columns
        ]
        plot_edge_lengths = [
            section_lengths[sim_name][section] for section in sections_to_plot
        ]
        plot_edge_num_pixels = np.round(np.array(plot_edge_lengths) / length_resolution)
        plot_edges_rounded_lengths = (
            np.round(np.array(plot_edge_lengths) / length_resolution)
            * length_resolution
            / 1000
        )
        for i, (edge, length) in enumerate(zip(sections_to_plot, plot_edge_num_pixels)):
            if i == 0:
                sim_data_per_meter = np.repeat(
                    sim_data[edge].values.reshape((-1, 1)), length, axis=1
                )
            else:
                sim_data_per_meter = np.concatenate(
                    [
                        sim_data_per_meter,
                        np.repeat(
                            sim_data[edge].values.reshape((-1, 1)), length, axis=1
                        ),
                    ],
                    axis=1,
                )

        total_length = sum(plot_edges_rounded_lengths)
        ax.imshow(
            sim_data_per_meter.T * data_scale,
            extent=(
                sim_data.index[0],
                sim_data.index[-1],
                0,
                total_length,
            ),
            aspect=(len(sim_data.index) / steps_per_second) / total_length,
            cmap=cmap,
            vmin=min_val,
            vmax=max_val,
            interpolation="none",
            origin="lower",
        )
        write_x_label = True
        write_y_label = True
        if not write_all_axis_labels:
            write_x_label = int(j / num_cols) == (num_rows - 1)
            write_y_label = j % num_cols == 0

        if write_x_label:
            ax.set_xlabel("Simulation time [s]", fontsize=axis_label_size)
        if write_y_label:
            ax.set_ylabel("Highway location [km]", fontsize=axis_label_size)
        ax.tick_params(axis="both", which="major", labelsize=tick_label_size)
        ax.set_title(fill(sim_name, max_subplot_title_len), fontsize=sim_title_size)

    for ax in axes.flatten()[len(data.keys()) :]:
        ax.set_axis_off()

    fig.set_figwidth(4.7 * num_cols)
    fig.set_figheight(4.9 * num_rows)
    # plt.yticks(fontsize="x-large")

    colorbar_ax: plt.Axes = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    colorbar_ax.tick_params(axis="both", which="major", labelsize=tick_label_size)

    if data_name is not None:
        colorbar_ax.set_title(
            fill(data_name, max_color_bar_title_len),
            loc="left",
            fontsize=axis_label_size,
        )

    norm = colors.Normalize(vmin=min_val, vmax=max_val)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_ax)
    # colorbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_ax)
    # if data_name is not None:
    #     colorbar.set_label(data_name)
    if title is not None:
        fig.suptitle(
            title,
            fontsize="xx-large",
            fontweight="bold",
        )
    return fig
