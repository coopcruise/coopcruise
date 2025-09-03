import argparse
import numpy as np
from pathlib import Path
from utils.analysis_utils import (
    get_sim_results_dir_nested,
    analyze_multiple_sim_groups,
    plot_mixed_stat_results,
    FIG_SAVE_FORMATS,
)

NUM_ROLLOUT_WORKERS = 10
INFLOW_TIME_HEADWAY = 2
AV_PERCENT = 100

SINGLE_LANE = False  # True
NUM_CONTROL_SEGMENTS = 2  # 3 # 4 # 5
PER_LANE_CONTROL = False  # True

NUM_SIMULATION_STEPS_PER_STEP = 5
SIMULATION_TIME = 500  # if SINGLE_LANE else 1000

SECONDS_PER_STEP = 0.5


USE_LIBSUMO = True
SHOW_GUI_IN_TRACI_MODE = True

# Set seed to an integer for deterministic simulation. Set to None for
# default behavior.
RANDOM_SEED = None
SUMO_SEED = None  # 0  # None
RANDOM_AV_SWITCHING_SEED = None  # 0  # None

NUM_TESTS = 30

RESULTS_DIR = "test"

CHANGE_LC_AV_ONLY = False
NO_LC = False
NO_LC_RIGHT = True
LC_PARAMS = (
    None  # dict(lcKeepRight=0, lcAssertive=2.5, lcSpeedGain=5, lcImpatience=0.7)
)
DEFAULT_TAU = None
RANDOM_AV_SWITCHING = True
HUMAN_SPEED_STD_0 = True

WARM_UP_TIME = 200  # sec
MERGE_FLOW_DURATION_SINGLE_LANE = 30  # sec
MERGE_FLOW_DURATION_MULTI_LANE = 50  # sec
BREAK_PERIOD_DURATION = 8400  # sec

KEEP_VEH_NAMES_NO_MERGE = True
# Requires creating a different flow file to change that


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Train area-based time-headway traffic congestion controller using PPO.",
        epilog="python3 -i <this-script>",
    )

    # optional input parameters

    parser.add_argument(
        "--av_percent",
        type=int,
        nargs="+",
        default=[AV_PERCENT],
        help=(
            "Percent of ACC-equipped vehicles. "
            "To compare between multiple simulations, specify more than one value here."
        ),
    )

    parser.add_argument(
        "--single_lane",
        default=SINGLE_LANE,
        action="store_true",
        help="Whether to use a single lane scenario",
    )

    parser.add_argument(
        "--num_tests",
        type=int,
        default=NUM_TESTS,
        help="Number of tests to run for each configuration.",
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        nargs="+",
        default=[RESULTS_DIR],
        help=(
            "Simulation results output directory name. "
            "To compare between multiple controllers, specify more than one value here."
        ),
    )

    # parser.add_argument(
    #     "--results_dir",
    #     type=str,
    #     default=RESULTS_DIR,
    #     help="Simulation results output directory name.",
    # )
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    num_tests = args.num_tests
    random_av_switching_seed = (
        list(np.arange(num_tests))  # 0 # None
    )

    use_tau_control = False  # [False, True]
    tau_control_only_rightmost_lane = False
    automatic_tau_duration = True

    single_lane = args.single_lane

    av_percent = args.av_percent

    results_dirs = args.results_dir

    compare_controllers = len(results_dirs) > 1
    performances = []

    for results_dir in results_dirs:
        scenario_dir = Path("scenarios/single_junction") / results_dir

        # network_file_name = None
        network_file_name = (
            "short_merge_lane_separate_exit_lane_disconnected_merge_lane.net.xml"
        )

        common_sim_params = {
            "scenario_dir": scenario_dir,
            "od_flow_file_name": "edge_flows_interval_8400_taz_reduced",
            "single_lane": single_lane,
            "change_lc_av_only": CHANGE_LC_AV_ONLY,
            "no_lc": NO_LC,
            "no_lc_right": NO_LC_RIGHT,
            "lc_params": LC_PARAMS,
            "warm_up_time": WARM_UP_TIME,
            "merge_flow_duration_single_lane": MERGE_FLOW_DURATION_SINGLE_LANE,
            "merge_flow_duration_multi_lane": MERGE_FLOW_DURATION_MULTI_LANE,
            "break_period_duration": BREAK_PERIOD_DURATION,
            "default_tau": DEFAULT_TAU,
            "keep_veh_names_no_merge": KEEP_VEH_NAMES_NO_MERGE,
            "inflow_time_headway": INFLOW_TIME_HEADWAY,
            "custom_name_postfix": None,
            # "use_learned_control": False,  # True,
            "tau_control_only_rightmost_lane": tau_control_only_rightmost_lane,
            "human_speed_std_0": HUMAN_SPEED_STD_0,
            "random_av_switching": RANDOM_AV_SWITCHING,
            "use_tau_control": use_tau_control,
            # "no_merge": False,
            # "random_av_switching_seed": 0,
            # "av_percent": AV_PERCENT,
        }

        compare_within_sim_params = {
            "no_merge": [True, False],
            "use_learned_control": [False, True],
            # "use_tau_control": [False, True],
            # "tau_val": [None] + list(np.arange(2, 6.5, 0.5)),
            # "tau_val": [None] + np.arange(1.6, 2.6, 0.1).round(2).tolist(),
        }

        compare_between_sim_params = {
            "av_percent": av_percent,
            "random_av_switching_seed": random_av_switching_seed,
        }

        multi_sim_result_dirs = get_sim_results_dir_nested(
            common_sim_params=common_sim_params,
            compare_within_sim_params=compare_within_sim_params,
            compare_between_sim_params=compare_between_sim_params,
        )

        multi_sim_group_name = "Single-Lane" if single_lane else "Multi-lane"
        save_dir = Path("results") / results_dir / multi_sim_group_name
        performance = analyze_multiple_sim_groups(multi_sim_result_dirs)
        performances.append(performance)

        fig_mixed_stat, ax_mixed_stat = plot_mixed_stat_results(
            mixed_stats=performance,
            title=multi_sim_group_name,
            std_scale_factor=1.96,
        )
        ax_mixed_stat.set_xlim(left=0 - 100 * 0.05, right=100 * 1.05)

        Path(save_dir).mkdir(exist_ok=True)
        for format in FIG_SAVE_FORMATS:
            fig_mixed_stat.savefig(
                Path(save_dir)
                / f"{multi_sim_group_name.lower().replace(' ', '_')}_performance.{format}"
            )

    if compare_controllers:
        compare_save_dir = (
            Path("results") / "_vs_".join(results_dirs) / multi_sim_group_name
        )

        ax_mixed_stat = None
        line_num = 0
        for performance, results_dir in zip(performances, results_dirs):
            fig_mixed_stat, ax_mixed_stat = plot_mixed_stat_results(
                mixed_stats=performance,
                title=multi_sim_group_name,
                std_scale_factor=1.96,
                label=results_dir,
                ax=ax_mixed_stat,
                line_num=line_num,
            )
            line_num += 1

        ax_mixed_stat.set_xlim(left=0 - 100 * 0.05, right=100 * 1.05)
        ax_mixed_stat.legend()

        Path(compare_save_dir).mkdir(parents=True, exist_ok=True)
        for format in FIG_SAVE_FORMATS:
            fig_mixed_stat.savefig(
                Path(compare_save_dir)
                / f"{multi_sim_group_name.lower().replace(' ', '_')}_performance.{format}"
            )
