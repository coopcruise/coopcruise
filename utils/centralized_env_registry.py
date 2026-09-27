"""Names and lookup for centralized SUMO RL environment classes."""

from sumo_centralized_envs_new import (
    SumoEnvCentralizedDistanceHeadway,
    SumoEnvCentralizedSpeedLimit,
    SumoEnvCentralizedTimeHeadway,
)

CANONICAL_ENV_CLASS_STRS = (
    "SumoEnvCentralizedTimeHeadway",
    "SumoEnvCentralizedDistanceHeadway",
    "SumoEnvCentralizedSpeedLimit",
)

LEGACY_ENV_CLASS_STRS = (
    "SumoEnvCentralizedTau",
    "SumoEnvCentralizedMinGap",
    "SumoEnvCentralizedVel",
)

ENV_CLS_STR_OPTIONS = list(CANONICAL_ENV_CLASS_STRS) + list(LEGACY_ENV_CLASS_STRS)

LEGACY_TO_CANONICAL_ENV_CLASS_STR = {
    "SumoEnvCentralizedTau": "SumoEnvCentralizedTimeHeadway",
    "SumoEnvCentralizedMinGap": "SumoEnvCentralizedDistanceHeadway",
    "SumoEnvCentralizedVel": "SumoEnvCentralizedSpeedLimit",
}

ENV_CLASS_PLOT_LABELS = {
    "SumoEnvCentralizedTimeHeadway": "TimeHeadway",
    "SumoEnvCentralizedDistanceHeadway": "DistanceHeadway",
    "SumoEnvCentralizedSpeedLimit": "SpeedLimit",
}

_ENV_CLASS_BY_STR = {
    "SumoEnvCentralizedTimeHeadway": SumoEnvCentralizedTimeHeadway,
    "SumoEnvCentralizedDistanceHeadway": SumoEnvCentralizedDistanceHeadway,
    "SumoEnvCentralizedSpeedLimit": SumoEnvCentralizedSpeedLimit,
}


def normalize_env_class_str(env_class_str: str) -> str:
    return LEGACY_TO_CANONICAL_ENV_CLASS_STR.get(env_class_str, env_class_str)


def get_env_class_from_str(env_class_str: str):
    canonical = normalize_env_class_str(env_class_str)
    env_class = _ENV_CLASS_BY_STR.get(canonical)
    if env_class is None:
        raise ValueError(
            f"env_class argument must be one of: {ENV_CLS_STR_OPTIONS}. "
            f"Got: {env_class_str!r}"
        )
    return env_class


def is_time_headway_env_class(env_class_str: str) -> bool:
    return (
        normalize_env_class_str(env_class_str) == "SumoEnvCentralizedTimeHeadway"
    )


def env_class_plot_label(env_class_str: str) -> str:
    canonical = normalize_env_class_str(env_class_str)
    return ENV_CLASS_PLOT_LABELS.get(canonical, env_class_str)


def results_dir_plot_label(results_dir: str) -> str:
    """Paper-style controller label from a results or scenario directory name."""
    if "_const_control" in results_dir and any(
        name in results_dir
        for name in (
            "SumoEnvCentralizedVel",
            "SumoEnvCentralizedSpeedLimit",
        )
    ):
        return "Traditional VSL"

    for legacy, canonical in LEGACY_TO_CANONICAL_ENV_CLASS_STR.items():
        if legacy in results_dir:
            return ENV_CLASS_PLOT_LABELS[canonical]

    for canonical, label in ENV_CLASS_PLOT_LABELS.items():
        if canonical in results_dir:
            return label

    return results_dir
