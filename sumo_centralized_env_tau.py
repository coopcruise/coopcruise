import numpy as np
from typing import Any, Dict
import pandas as pd
from pathlib import Path
from gymnasium import spaces
import traci.constants as tc
from sumo_multi_agent_env import SumoEnv
from utils.sumo_utils import get_edge_length
from utils.global_utils import get_segment_num_vehicles, get_segment_veh_ids
from utils.sumo_utils import (
    get_route_file_path,
    extract_num_departed_veh_from_routes,
    get_veh_type_param,
)
from gymnasium.spaces import flatten_space, flatten


class SumoEnvCentralizedTau(SumoEnv):
    CENTRALIZED_AGENT_NAME = "centralized"
    DEF_NUM_SIMULATION_STEPS_PER_STEP = 1
    REWARD_TYPES = ["avg_speed", "outflow", "completion_time", "time_delay"]
    CUMULATIVE_REWARDS = ["outflow", "completion_time", "time_delay"]
    DEFAULT_REWARD_TYPE = "avg_speed"
    RETURN_TO_DEFAULT_CMD = -1
    DEF_MIN_TAU = 1.5
    DEF_MAX_TAU = 6  # 10

    def __init__(self, config):
        super().__init__(config)
        self.num_simulation_steps_per_step = int(
            config.get(
                "num_simulation_steps_per_step", self.DEF_NUM_SIMULATION_STEPS_PER_STEP
            )
        )
        # Assumes segment_data contains data of control segments, and highway
        # state segments.
        self.control_segments = (
            config.get("control_segments")
            if config.get("control_segments") is not None
            else [
                segment
                for segment, data in self.segment_data.items()
                if data["end"]["edge"] in self.control_edges
            ]
        )

        self.num_control_segments = (
            config.get("num_control_segments")
            if config.get("num_control_segments") is not None
            else len(self.control_segments)
        )

        self.control_segments = self.control_segments[-self.num_control_segments :]

        self.control_segment_idx = [
            list(self.segment_data.keys()).index(segment)
            for segment in self.control_segments
        ]

        self.highway_state_edges = (
            config.get("highway_state_edges")
            if config.get("highway_state_edges") is not None
            else self.highway_sorted_road_edges
        )
        self.highway_state_segments = (
            config.get(
                "highway_state_segments",
            )
            if config.get(
                "highway_state_segments",
            )
            is not None
            else [
                segment
                for segment, data in self.segment_data.items()
                if data["end"]["edge"] in self.highway_state_edges
            ]
        )
        self.num_highway_state_segments = len(self.highway_state_segments)
        self.highway_state_segment_idx = [
            list(self.segment_data.keys()).index(segment)
            for segment in self.highway_state_segments
        ]

        self.is_single_lane = (
            config.get("is_single_lane")
            if config.get("is_single_lane") is not None
            else False
        )

        self.min_tau = (
            config.get("min_tau")
            if config.get("min_tau") is not None
            else self.DEF_MIN_TAU
        )
        self.max_tau = (
            config.get("max_tau")
            if config.get("max_tau") is not None
            else self.DEF_MAX_TAU
        )

        self.per_lane_control = (
            config.get("per_lane_control")
            if config.get("per_lane_control") is not None
            else False
        ) and not self.is_single_lane

        self.action_space_len = self.num_control_segments
        self.control_segment_lanes = self.num_segment_lanes[self.control_segment_idx]
        self.num_control_segment_lanes = sum(self.control_segment_lanes)
        self.cumsum_control_segment_lanes = np.cumsum(self.control_segment_lanes)
        if self.per_lane_control:
            self.action_space_len = self.num_control_segment_lanes

        self.action_space = spaces.Box(
            float(0), float(1), shape=(self.action_space_len,), dtype=np.float32
        )

        self.route_path = get_route_file_path(self.sumo_config_path)
        self.default_tau = get_veh_type_param(
            self.route_path, self.control_veh_type, "tau"
        )
        self._update_required_tau_profile()

        self.segment_max_speed = np.array(
            [
                self.edge_max_speed[segment["end"]["edge"]]
                for segment in self.segment_data.values()
            ]
        )
        self.segment_max_speed_norm = (
            self.segment_max_speed / self.normalization_max_speed
        )

        self._update_required_speed_profile()

        # self.prev_step_veh_ids = set()
        self.use_outflow_reward = (
            config.get("use_outflow_reward")
            if config.get("use_outflow_reward") is not None
            else False
        )
        self.use_completion_time_reward = (
            config.get("use_completion_time_reward")
            if config.get("use_completion_time_reward") is not None
            else False
        )
        self.use_time_delay_reward = (
            config.get("use_time_delay_reward")
            if config.get("use_time_delay_reward") is not None
            else False
        )

        if self.use_outflow_reward and self.use_completion_time_reward:
            raise ValueError(
                "Only one type of reward can be used. "
                "use_outflow_reward and use_time_delay_reward cannot both be True"
            )

        self.reward_type = self.DEFAULT_REWARD_TYPE
        self.get_reward_func = self._get_centralized_speed_reward
        self.cum_departures = extract_num_departed_veh_from_routes(
            self.route_path,
            end_time=self.simulation_time,
            seconds_per_step=self.sumo_config.seconds_per_step,
        )
        if self.use_outflow_reward:
            self.reward_type = "outflow"
            self.get_reward_func = self._get_centralized_outflow_reward
        elif self.use_completion_time_reward:
            self.reward_type = "completion_time"
            self.get_reward_func = self._get_centralized_completion_time_reward
        elif self.use_time_delay_reward:
            self.reward_type = "time_delay"
            self.get_reward_func = self._get_centralized_time_delay_reward
            # self.max_num_veh = extract_num_veh_from_routes(
            # self.num_completed_veh = 0

        # TODO: Refactor segment creation and extraction to make the following simpler
        self.state_merge_edges: list = (
            config.get("state_merge_edges")
            if config.get("state_merge_edges") is not None
            else []
        )
        self.num_merge_edges = len(self.state_merge_edges)
        self.edge_lengths = get_edge_length(self.network_path)
        self.merge_lengths = [
            self.edge_lengths[merge_id] for merge_id in self.state_merge_edges
        ]
        self.merge_lengths = np.array(self.merge_lengths, dtype=np.float32)
        # TODO: Add merge locations along road.

        self.include_tse_pos_in_obs = (
            config.get("include_tse_pos_in_obs")
            if config.get("include_tse_pos_in_obs") is not None
            else True
        )
        self.include_av_frac_in_obs = (
            config.get("include_av_frac_in_obs")
            if config.get("include_av_frac_in_obs") is not None
            else True
        )

        self.flat_obs_space = (
            config.get("flat_obs_space")
            if config.get("flat_obs_space") is not None
            else False
        )

        self.observation_space = self.get_centralized_obs_space(
            self.num_highway_state_segments,
            self.num_merge_edges,
            self.include_tse_pos_in_obs,
            self.include_av_frac_in_obs,
        )

        if self.flat_obs_space:
            self.original_obs_space = self.observation_space
            self.observation_space = flatten_space(self.observation_space)

        self.normalization_max_density = 1 / (4 * self.normalization_veh_len)
        self.max_measurement_location = max(self.measurement_locations)

    @staticmethod
    def get_centralized_obs_space(
        num_highway_state_segments,
        num_merge_edges=0,
        include_tse_pos_in_obs=True,
        include_av_frac_in_obs=True,
    ):
        obs_space = {
            "tse_speed": spaces.Box(
                float(0),
                float("inf"),
                shape=(num_highway_state_segments,),
                dtype=np.float32,
            ),
            "tse_density": spaces.Box(
                float(0),
                float("inf"),
                shape=(num_highway_state_segments,),
                dtype=np.float32,
            ),
        }
        if include_tse_pos_in_obs:
            obs_space.update(
                {
                    "tse_pos": spaces.Box(
                        float(0),
                        float("inf"),
                        shape=(num_highway_state_segments,),
                        dtype=np.float32,
                    )
                }
            )

        if include_av_frac_in_obs:
            obs_space.update(
                {
                    "tse_av_fraction": spaces.Box(
                        float(0),
                        float(1),
                        shape=(num_highway_state_segments,),
                        dtype=np.float32,
                    )
                }
            )

        if num_merge_edges > 0:
            obs_space.update(
                {
                    "tse_merge_speed": spaces.Box(
                        float(0),
                        float("inf"),
                        shape=(num_merge_edges,),
                        dtype=np.float32,
                    ),
                    "tse_merge_density": spaces.Box(
                        float(0),
                        float("inf"),
                        shape=(num_merge_edges,),
                        dtype=np.float32,
                    ),
                }
            )

        return spaces.Dict(obs_space)

    def reset(self, *args, seed=None, options=None):
        super().reset(*args, seed=seed, options=options)
        self._update_required_speed_profile()
        self._update_required_tau_profile()
        self.num_waiting_veh = 0
        self._agent_ids = set([self.CENTRALIZED_AGENT_NAME])
        veh_data = self._get_veh_data()
        centralized_obs = self._get_centralized_obs(veh_data)
        centralized_info = {}
        self.prev_step_veh_ids = set(veh_data.keys())
        self.current_reward = {self.CENTRALIZED_AGENT_NAME: 0}
        return centralized_obs, centralized_info

    def step(self, action: Dict[str, Any]):
        if self.CENTRALIZED_AGENT_NAME in action.keys():
            self._update_required_tau_profile(action[self.CENTRALIZED_AGENT_NAME])

        # Extract AV tau actions from tau profile
        av_tau_actions = self._extract_av_tau_actions()
        # Progress one step. Uses internal car following model
        # and progresses a single simulation step, returning speed profile
        # observations.
        self._send_tau_actions(av_tau_actions)
        obs, rewards, terminateds, truncateds, infos = super().step({})

        veh_data = self._get_veh_data()

        centralized_rewards = self.get_reward_func(
            prev_veh_ids=self.prev_step_veh_ids,
            current_veh_ids=self.current_veh_ids,
            current_time_step=self.step_count,
            veh_data=veh_data,
        )

        if self.reward_type in self.CUMULATIVE_REWARDS:
            self.current_reward[self.CENTRALIZED_AGENT_NAME] += centralized_rewards[
                self.CENTRALIZED_AGENT_NAME
            ]
        else:
            self.current_reward[self.CENTRALIZED_AGENT_NAME] = centralized_rewards[
                self.CENTRALIZED_AGENT_NAME
            ]

        self.num_waiting_veh = (
            self.cum_departures[min(self.step_count, len(self.cum_departures) - 1)]
            - self.num_completed_veh
            - len(self.current_veh_ids)
        )
        # Return observation either when the episode ends or when the speed
        # profile should be updated.
        if (
            truncateds["__all__"]
            or terminateds["__all__"]
            or self.step_count % self.num_simulation_steps_per_step == 0
        ):
            centralized_obs = self._get_centralized_obs(veh_data)
            self._agent_ids = set([self.CENTRALIZED_AGENT_NAME])
            # print("timestep", self.step_count, "reward", self.current_reward)
            centralized_terminateds = {"__all__": terminateds["__all__"]}
            centralized_truncateds = {"__all__": truncateds["__all__"]}
            reward = self.current_reward.copy()
            self.current_reward = {self.CENTRALIZED_AGENT_NAME: 0}
            return (
                centralized_obs,
                reward,
                centralized_terminateds,
                centralized_truncateds,
                infos,
            )

        # Recursively progress the simulation until the profile should be updated
        return self.step({})

    def _update_required_tau_profile(self, required_tau_norm=None):
        if required_tau_norm is None:
            # Initialize required tau profile using default value for AV.
            self.required_tau_profile = (
                np.ones(self.action_space_len) * self.default_tau
            )
        else:
            # Map 1 to the max speed of the segment
            self.required_tau_profile = (
                required_tau_norm * (self.max_tau - self.min_tau) + self.min_tau
            )

    def _update_required_speed_profile(self, required_speed_profile_norm=None):
        # Only use segments in control edges!
        if required_speed_profile_norm is None:
            # Initialize required speed profile using edge speed limits.
            self.required_speed_profile = self.segment_max_speed[
                self.control_segment_idx
            ]
        else:
            # Map 1 to the max speed of the segment
            self.required_speed_profile = (
                required_speed_profile_norm
                * self.segment_max_speed[self.control_segment_idx]
            )

    def _extract_av_tau_actions(self):
        av_data = self._get_veh_data(self.control_veh_type)
        # Assumes control segments are a subset of the state segments.
        av_segment_ids = get_segment_veh_ids(
            av_data, self.edge_segment_map, self.segment_data
        )
        controlled_av_segment_ids = {
            segment_id: segment_avs if segment_id in self.control_segments else []
            for segment_id, segment_avs in av_segment_ids.items()
        }
        # Return all non controlled AVs to regular road speed:
        av_tau_actions = {av_id: self.RETURN_TO_DEFAULT_CMD for av_id in av_data.keys()}
        for segment_idx, segment_avs in enumerate(controlled_av_segment_ids.values()):
            # The following condition should always be correct, since we extract
            # only AVs in controlled edges.
            if segment_idx in self.control_segment_idx:
                tau_prof_segment_idx = self.control_segment_idx.index(segment_idx)
                tau_prof_idx = tau_prof_segment_idx * np.ones(len(segment_avs), int)
                if self.per_lane_control:
                    num_lanes_prev_segments = (
                        0
                        if tau_prof_segment_idx == 0
                        else self.cumsum_control_segment_lanes[tau_prof_segment_idx - 1]
                    )
                    tau_prof_idx = num_lanes_prev_segments + np.array(
                        [
                            int(av_data[av_id][tc.VAR_LANE_ID].split("_")[-1])
                            for av_id in segment_avs
                        ]
                    )

                av_tau_actions.update(
                    {
                        av_id: self.required_tau_profile[idx]
                        for av_id, idx in zip(segment_avs, tau_prof_idx)
                    }
                )
        return av_tau_actions

    def _send_tau_actions(self, av_tau_actions: dict):
        veh_data = self._get_veh_data()
        for agent_id, required_tau in av_tau_actions.items():
            if required_tau < 0:
                if not veh_data[agent_id][tc.VAR_TYPE] == self.control_veh_type:
                    self.traci_conn.vehicle.setType(agent_id, self.control_veh_type)
            else:
                self.traci_conn.vehicle.setTau(agent_id, required_tau.astype(float))

    def _extract_av_speed_actions(self):
        av_data, controlled_av_data = self._get_av_data()
        # Assumes control segments are a subset of the state segments.
        controlled_av_segment_ids = get_segment_veh_ids(
            controlled_av_data, self.edge_segment_map, self.segment_data
        )
        # Return all non controlled AVs to regular road speed:
        av_speed_actions = {
            av_id: self.RETURN_TO_DEFAULT_CMD for av_id in av_data.keys()
        }
        for idx, av_list in enumerate(controlled_av_segment_ids.values()):
            # The following condition should always be correct, since we extract
            # only AVs in controlled edges.
            if idx in self.control_segment_idx:
                speed_prof_idx = self.control_segment_idx.index(idx)
                av_speed_actions.update(
                    {
                        # Normalize speed actions
                        av_id: self.required_tau_profile[speed_prof_idx]
                        / self.normalization_max_speed
                        for av_id in av_list
                    }
                )
        return av_speed_actions

    def _get_centralized_obs(self, veh_data: dict):
        num_highway_state_segment_veh = np.array(
            list(
                get_segment_num_vehicles(
                    veh_data, self.edge_segment_map, self.segment_data
                ).values()
            )
        )[self.highway_state_segment_idx]
        num_highway_state_segment_avs = np.array(
            list(
                get_segment_num_vehicles(
                    veh_data,
                    self.edge_segment_map,
                    self.segment_data,
                    self.control_veh_type,
                ).values()
            )
        )

        # For segments with 0 vehicles, use the required speed profile (from the
        # latest actions). segment_speed is not normalized.
        segment_speed = np.array(self.speed_measurements)
        segment_speed[self.control_segment_idx] = np.where(
            num_highway_state_segment_avs[self.control_segment_idx] == 0,
            self.required_speed_profile,
            segment_speed[self.control_segment_idx],
        )
        highway_state_segment_speed = segment_speed[self.highway_state_segment_idx]

        state = {
            self.CENTRALIZED_AGENT_NAME: {
                "tse_speed": highway_state_segment_speed.astype(np.float32)
                / self.normalization_max_speed,
                # Single lane scaling - no need to divide by number of lanes for single lane scenario
                "tse_density": (
                    num_highway_state_segment_veh
                    / self.segment_lengths[self.highway_state_segment_idx]
                    / (
                        1
                        if self.is_single_lane
                        else self.num_segment_lanes[self.highway_state_segment_idx]
                    )
                    / self.normalization_max_density
                ).astype(np.float32),
            }
        }
        if self.include_tse_pos_in_obs:
            state.update(
                {
                    # Scale this by the maximum segment end location. Needed only if training on many difference scenarios
                    "tse_pos": np.array(
                        self.measurement_locations[self.highway_state_segment_idx]
                    ).astype(np.float32)
                    / self.max_measurement_location,
                }
            )

        if self.include_av_frac_in_obs:
            state.update(
                {
                    "tse_av_fraction": np.divide(
                        num_highway_state_segment_avs,
                        num_highway_state_segment_veh,
                        where=num_highway_state_segment_veh != 0,
                        out=np.zeros(
                            num_highway_state_segment_avs.shape,
                            dtype=np.float32,
                        ),
                        dtype=np.float32,
                    ),
                }
            )

        # TODO: Make this simpler:
        if self.num_merge_edges > 0:
            num_merge_vehicles = np.zeros(self.num_merge_edges, dtype=np.float32)
            merge_speed = {merge_id: [] for merge_id in self.state_merge_edges}
            for vehicle_params in veh_data.values():
                edge = vehicle_params[tc.VAR_ROAD_ID]
                if edge in self.state_merge_edges:
                    merge_idx = self.state_merge_edges.index(edge)
                    num_merge_vehicles[merge_idx] += 1
                    merge_speed[edge].append(vehicle_params[tc.VAR_SPEED])
            merge_avg_speed = np.array(
                [
                    np.average(speeds)
                    if len(speeds) > 0
                    else self.edge_max_speed[merge_id]
                    for merge_id, speeds in merge_speed.items()
                ],
                dtype=np.float32,
            )
            merge_density = num_merge_vehicles / self.merge_lengths
            state[self.CENTRALIZED_AGENT_NAME].update(
                {
                    "tse_merge_speed": merge_avg_speed / self.normalization_max_speed,
                    "tse_merge_density": merge_density / self.normalization_max_density,
                }
            )

        if self.flat_obs_space:
            state[self.CENTRALIZED_AGENT_NAME] = flatten(
                self.original_obs_space, state[self.CENTRALIZED_AGENT_NAME]
            )

        return state

    def _get_centralized_speed_reward(self, *args, **kwargs):
        # Compute centralized average speed rewards. As a starting point, use
        # current average speed, computed similarly to the cooperative reward in
        # self._get_speed_rewards()
        max_speed = self.normalization_max_speed
        avg_speed = (
            self.segment_lengths.sum()
            # Assume piecewise constant speed
            / (self.segment_lengths / np.array(self.speed_measurements)).sum()
        )
        return {self.CENTRALIZED_AGENT_NAME: avg_speed / max_speed}

    def _get_centralized_outflow_reward(
        self, prev_veh_ids, current_veh_ids, *args, **kwargs
    ):
        num_exit_veh = len(set(prev_veh_ids) - set(current_veh_ids))
        return {self.CENTRALIZED_AGENT_NAME: num_exit_veh}

    def _get_centralized_completion_time_reward(
        self, current_time_step, *args, **kwargs
    ):
        return {
            self.CENTRALIZED_AGENT_NAME: -self.sumo_config.seconds_per_step
            * (
                self.cum_departures[
                    min(current_time_step, len(self.cum_departures) - 1)
                ]
                - self.num_completed_veh
            )
            / self.cum_departures[-1]
        }

    def _get_centralized_time_delay_reward(
        self,
        current_time_step,
        veh_data,
        *args,
        **kwargs,
    ):
        sum_speed_ratios = np.sum(
            [
                vehicle_params[tc.VAR_SPEED]
                / self.edge_max_speed[vehicle_params[tc.VAR_ROAD_ID]]
                for vehicle_params in veh_data.values()
            ]
        )
        return {
            self.CENTRALIZED_AGENT_NAME: -self.sumo_config.seconds_per_step
            * (
                self.cum_departures[
                    min(current_time_step, len(self.cum_departures) - 1)
                ]
                - self.num_completed_veh
                - sum_speed_ratios
            )
            / self.cum_departures[-1]
            * 0.01  # scaling so that values are around 1
        }

    # TODO: Store Tau actions
    def _initialize_metrics_data_structures(self):
        super()._initialize_metrics_data_structures()
        self.required_segment_speed_limit_actions = pd.DataFrame(
            np.zeros((self.num_steps, len(self.segment_data))),
            index=np.arange(self.num_steps) * self.sumo_config.seconds_per_step,
            columns=self.segment_data.keys(),
        )

    def _update_metrics(self):
        super()._update_metrics()
        segment_speed_limits = self.segment_max_speed.copy()
        segment_speed_limits[self.control_segment_idx] = self.required_speed_profile
        self.required_segment_speed_limit_actions.iloc[self.step_count] = (
            segment_speed_limits
        )

    def log_episode(self):
        super().log_episode()
        if self.eval_config is not None:
            self.required_segment_speed_limit_actions.to_csv(
                Path(self.episode_results_dir) / "segment_speed_limit.csv"
            )
