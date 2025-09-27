import argparse
from pathlib import Path
from evaluate_control_rl import MERGE_FLOW_PERCENT
from utils.analysis_utils import (
    get_sim_results_dir_nested,
    analyze_multiple_sim_groups,
    plot_mixed_stat_results,
    FIG_SAVE_FORMATS,
)
from train_ppo_centralized import (
    INFLOW_TIME_HEADWAY,
    AV_PERCENT,
    SINGLE_LANE,
    CHANGE_LC_AV_ONLY,
    NO_LC,
    NO_LC_RIGHT,
    LC_PARAMS,
    DEFAULT_TAU,
    RANDOM_AV_SWITCHING,
    HUMAN_SPEED_STD_0,
    WARM_UP_TIME,
    MERGE_FLOW_DURATION_SINGLE_LANE,
    MERGE_FLOW_DURATION_MULTI_LANE,
    BREAK_PERIOD_DURATION,
    KEEP_VEH_NAMES_NO_MERGE,
)
from evaluate_control_new import RESULTS_DIR, NUM_TESTS

SCENARIO_DIR = "scenarios/single_junction"
DEF_SIM_PARAMS = {
    "od_flow_file_name": "edge_flows_interval_8400_taz_reduced",
    "change_lc_av_only": CHANGE_LC_AV_ONLY,
    "warm_up_time": WARM_UP_TIME,
    "no_lc": NO_LC,
    "no_lc_right": NO_LC_RIGHT,
    "lc_params": LC_PARAMS,
    "merge_flow_duration_single_lane": MERGE_FLOW_DURATION_SINGLE_LANE,
    "merge_flow_duration_multi_lane": MERGE_FLOW_DURATION_MULTI_LANE,
    "break_period_duration": BREAK_PERIOD_DURATION,
    "default_tau": DEFAULT_TAU,
    "keep_veh_names_no_merge": KEEP_VEH_NAMES_NO_MERGE,
    "inflow_time_headway": INFLOW_TIME_HEADWAY,
    "custom_name_postfix": None,
    # "use_learned_control": False,  # True,
    "human_speed_std_0": HUMAN_SPEED_STD_0,
    "random_av_switching": RANDOM_AV_SWITCHING,
}


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Train area-based time-headway traffic congestion controller using PPO.",
        epilog="python3 -i <this-script>",
    )

    # optional input parameters
    # TODO: Extract all parameters from directory names within the results directory
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


def extract_params(results_dir: str):
    scenario_dir = Path(SCENARIO_DIR) / results_dir
    results_dir_path = Path("results") / results_dir

    # network_file_name = None
    # network_file_name = (
    #     "short_merge_lane_separate_exit_lane_disconnected_merge_lane.net.xml"
    # )
    if not results_dir_path.exists():
        raise ValueError(f"results dir: {results_dir_path} does not exist!")

    result_dir_name_str = "edge_flows_interval"
    subdirs = [
        x.name
        for x in Path(results_dir_path).iterdir()
        if x.is_dir() and result_dir_name_str in x.name
    ]

    params = {
        "warm_up_time": set(
            [
                int(subdir.split("warmup_")[1].split("_")[0])
                for subdir in subdirs
                if "warmup" in subdir
            ]
        ),
        "merge_flow_percent": set(
            [
                int(subdir.split("merge_flow_percent_")[1].split("_")[0])
                if "merge_flow_percent" in subdir
                else MERGE_FLOW_PERCENT
                for subdir in subdirs
            ]
        ),
        "human_speed_std_0": set(["human_speed_std_0" in subdir for subdir in subdirs]),
        "no_lc_right": set(["no_lc_right" in subdir for subdir in subdirs]),
        "av_percent": set(
            [
                int(subdir.split("av_percent_")[1].split("_")[0])
                for subdir in subdirs
                if "av_percent" in subdir
            ]
        ),
        "use_learned_control": set(["rl_control" in subdir for subdir in subdirs]),
        "random_av_switching_seed": set(
            [
                int(subdir.split("av_switch_seed_")[1].split("_")[0])
                for subdir in subdirs
                if "av_switch_seed" in subdir
            ]
        ),
        # "no_merge": set(["no_merge" in subdir for subdir in subdirs]),
        "no_merge": [True, False],  # Must be exactly this
        "single_lane": set(["single_lane" in subdir for subdir in subdirs]),
        # "per_lane_control": set(["per_lane" in subdir for subdir in subdirs]),
        # "right_lane_control": set(["right_lane" in subdir for subdir in subdirs]),
        "use_tau_control": set(["tau_control" in subdir for subdir in subdirs]),
        "tau_val": set(
            [None]
            + [
                float(subdir.split("tau_control_")[1].split("_")[0])
                for subdir in subdirs
                if "tau_control" in subdir
            ]
        ),
        "tau_control_only_rightmost_lane": set(
            ["tau_control" in subdir and "rightmost" in subdir for subdir in subdirs]
        ),
    }

    params = {
        key: sorted(list(val))
        if len(val) > 1
        else (val.pop() if len(val) > 0 else None)
        for key, val in params.items()
    }

    compare_within_sim_params = {
        key: val
        for key, val in params.items()
        if isinstance(val, list)
        and len(val) > 1
        and key in ["no_merge", "use_learned_control", "use_tau_control", "tau_val"]
    }

    compare_between_sim_params = {
        key: (val if isinstance(val, list) else [val])
        for key, val in params.items()
        if val is not None and key in ["av_percent", "random_av_switching_seed"]
    }

    divide_keys = [
        key
        for key, val in params.items()
        if isinstance(val, list)
        and not any(
            [
                key in container
                for container in [compare_within_sim_params, compare_between_sim_params]
            ]
        )
    ]

    common_sim_params = (
        DEF_SIM_PARAMS
        | {"scenario_dir": scenario_dir}
        | {
            key: val
            for key, val in params.items()
            if not any(
                [
                    key in container
                    for container in [
                        compare_within_sim_params,
                        compare_between_sim_params,
                        divide_keys,
                    ]
                ]
            )
        }
    )

    # common_sim_params = {
    #     "scenario_dir": scenario_dir,
    #     "od_flow_file_name": "edge_flows_interval_8400_taz_reduced",
    #     "single_lane": single_lane,
    #     "change_lc_av_only": CHANGE_LC_AV_ONLY,
    #     "no_lc": NO_LC,
    #     "no_lc_right": NO_LC_RIGHT,
    #     "lc_params": LC_PARAMS,
    #     "merge_flow_duration_single_lane": MERGE_FLOW_DURATION_SINGLE_LANE,
    #     "merge_flow_duration_multi_lane": MERGE_FLOW_DURATION_MULTI_LANE,
    #     "break_period_duration": BREAK_PERIOD_DURATION,
    #     "default_tau": DEFAULT_TAU,
    #     "keep_veh_names_no_merge": KEEP_VEH_NAMES_NO_MERGE,
    #     "inflow_time_headway": INFLOW_TIME_HEADWAY,
    #     "custom_name_postfix": None,
    #     "use_learned_control": False,  # True,
    #     "tau_control_only_rightmost_lane": tau_control_only_rightmost_lane,
    #     "human_speed_std_0": HUMAN_SPEED_STD_0,
    #     "random_av_switching": RANDOM_AV_SWITCHING,
    #     "use_tau_control": use_tau_control,
    #     # "no_merge": False,
    #     # "random_av_switching_seed": 0,
    #     # "av_percent": AV_PERCENT,
    # }

    # compare_within_sim_params = {
    #     "no_merge": [True, False],
    #     # "use_learned_control": [False, True],
    #     "use_learned_control": list(set(["rl_control" in subdir for subdir in subdirs])),
    #     # "use_tau_control": [False, True],
    #     # "tau_val": [None] + list(np.arange(2, 6.5, 0.5)),
    #     # "tau_val": [None] + np.arange(1.6, 2.6, 0.1).round(2).tolist(),
    # }

    # compare_between_sim_params = {
    #     "av_percent": av_percent,
    #     "random_av_switching_seed": random_av_switching_seed,
    # }

    return (
        common_sim_params,
        compare_within_sim_params,
        compare_between_sim_params,
        divide_keys,
    )


def main():
    parser = create_parser()
    args = parser.parse_args()

    # num_tests = args.num_tests
    # random_av_switching_seed = (
    #     list(np.arange(num_tests))  # 0 # None
    # )

    # use_tau_control = False  # [False, True]
    # tau_control_only_rightmost_lane = False
    # # automatic_tau_duration = True

    # single_lane = args.single_lane

    # av_percent = args.av_percent

    results_dirs = args.results_dir

    compare_controllers = len(results_dirs) > 1
    performances = []

    for results_dir in results_dirs:
        (
            common_sim_params,
            compare_within_sim_params,
            compare_between_sim_params,
            divide_keys,  # TODO
        ) = extract_params(results_dir)
        # print(f"{common_sim_params = }")
        # print(f"{compare_within_sim_params = }")
        # print(f"{compare_between_sim_params = }")
        # print(f"{divide_keys = }")
        multi_sim_result_dirs = get_sim_results_dir_nested(
            common_sim_params=common_sim_params,
            compare_within_sim_params=compare_within_sim_params,
            compare_between_sim_params=compare_between_sim_params,
        )

        multi_sim_group_name = (
            "Single-Lane" if common_sim_params["single_lane"] else "Multi-lane"
        )
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


if __name__ == "__main__":
    main()
