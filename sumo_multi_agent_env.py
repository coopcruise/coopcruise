import uuid
import os
import json
import time

# import random
import pandas as pd
import numpy as np

# TODO: Use logging instead of prints
# import logging
from tqdm.auto import tqdm
import shutil
import traci.constants as tc
from typing import Any, Dict, List
from pathlib import Path
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from traci.exceptions import FatalTraCIError
from ray.rllib.utils.typing import MultiAgentDict

from utils.i24_utils import get_main_road_west_edges
from utils.metrics_utils import (
    step_count_from_interval_count,
    compute_throughput,
)
from utils.sumo_utils import (
    extract_vehicle_ids_from_routes,
    get_network_file_path,
    get_edge_max_speed,
    get_edge_ids,
    get_junction_ids,
    get_detector_ids,
    edge_distance_from_start,
    object_path_location,
    get_detector_results_files,
    detector_group_mapping,
    get_detector_segment_data,
    get_edge_segment_map,
    get_edge_num_lanes,
    extract_vehicle_departure_time_from_routes,
    get_route_file_path,
    extract_num_veh_per_type_from_route,
    get_episode_results_dir,
)

from utils.global_utils import (
    get_interval_counts,
    get_edge_veh_count_and_average_speed,
    get_lane_flows,
    get_vehicle_data,
    get_speed_profile,
    get_segment_avg_speed,
    get_segment_num_vehicles,
)

SECONDS_PER_HOUR = 3600


class EvaluationConfig:
    DEF_THROUGHPUT_LOOP_DETECTOR_INTERVAL_DURATION = 60  # seconds
    DEF_THROUGHPUT_COMPUTATION_PERIOD_TIME = 300
    DEF_THROUGHPUT_DETECTOR_FILE = "throughput_detectors.xml"

    def __init__(
        self,
        save_lane_flows: bool = False,
        throughput_detector_file=None,
        throughput_loop_detector_interval_duration=DEF_THROUGHPUT_LOOP_DETECTOR_INTERVAL_DURATION,
        throughput_computation_period_time=DEF_THROUGHPUT_COMPUTATION_PERIOD_TIME,
        save_throughput_for_speed_profile_detectors: bool = True,
        save_segment_data: bool = True,
    ):
        self.save_lane_flows = save_lane_flows
        self.throughput_detector_file = (
            self.DEF_THROUGHPUT_DETECTOR_FILE
            if throughput_detector_file is None
            else throughput_detector_file
        )
        self.throughput_loop_detector_interval_duration = (
            throughput_loop_detector_interval_duration
        )
        self.throughput_computation_period_time = throughput_computation_period_time
        self.save_throughput_for_speed_profile_detectors = (
            save_throughput_for_speed_profile_detectors
        )
        self.save_segment_data = save_segment_data


class SumoConfig:
    DEF_SECONDS_PER_STEP = 1
    DEF_NUM_SUB_LANES = 8
    DEF_LANE_WIDTH = 3.2
    DEF_STATE_DIR = "states"
    DEF_SAVE_STATE_TIME_INTERVALS = 150
    DEF_SPEED_PROFILE_DETECTOR_FILE = "add_highway_profile_detectors.xml"
    DEF_SPEED_PROFILE_DETECTOR_INTERVAL_DURATION = 60  # seconds

    def __init__(
        self,
        scenario_dir,
        sumo_config_file,
        save_states: bool = False,
        state_dir=None,
        save_state_time_intervals: int = DEF_SAVE_STATE_TIME_INTERVALS,
        load_state_file=None,
        seconds_per_step: float = DEF_SECONDS_PER_STEP,
        num_sub_lanes: int = DEF_NUM_SUB_LANES,
        show_gui: bool = True,
        lane_width: float = DEF_LANE_WIDTH,
        speed_profile_detector_interval_duration=DEF_SPEED_PROFILE_DETECTOR_INTERVAL_DURATION,
        speed_profile_detector_file=None,
        use_vehicle_based_speed_profile_update=True,
        no_warnings: bool = False,
        seed: int = None,
        use_libsumo: bool = False,
    ):
        self.scenario_dir = scenario_dir
        self.sumo_config_file = sumo_config_file
        self.state_dir = self.DEF_STATE_DIR if state_dir is None else state_dir
        self.save_states = save_states
        self.save_state_time_intervals = save_state_time_intervals
        self.load_state_file = load_state_file
        self.seconds_per_step = seconds_per_step
        self.num_sub_lanes = num_sub_lanes
        self.show_gui = show_gui
        self.lane_width = lane_width
        self.speed_profile_detector_interval_duration = (
            speed_profile_detector_interval_duration
        )
        self.speed_profile_detector_file = (
            self.DEF_SPEED_PROFILE_DETECTOR_FILE
            if speed_profile_detector_file is None
            else speed_profile_detector_file
        )
        self.use_vehicle_based_speed_profile_update = (
            use_vehicle_based_speed_profile_update
        )
        self.no_warnings = no_warnings
        self.seed = seed
        self.use_libsumo = use_libsumo


class SumoEnv(MultiAgentEnv):
    DEF_RESULTS_DIR = "results"
    DEF_CONTROL_VEH_TYPE = "AV"
    DEF_COMPLETED_REWARD = 20
    # This parameter defines the maximum lookahead for the leader, 0 calculates
    # a lookahead from the brake gap.
    DEF_MAX_LEADER_LOOKAHEAD = 300.0  # m
    DEF_VEH_LEN = 5.0  # m
    EPS = 1e-6
    action_space = spaces.Box(-float("inf"), float("inf"), shape=(1,), dtype=np.float32)

    def __init__(self, config: dict):
        super().__init__()
        # Initialize class arguments
        # self._agent_ids = set()
        # self._leader_subscribed_avs = set()
        self.simulation_time = config["simulation_time"]
        self.sumo_config: SumoConfig = config["sumo_config"]
        self.warm_up_time = config.get("warm_up_time", 0)
        control_veh_type = config.get("control_veh_type")
        # TODO: Move outside to the control policy
        self.control_edges: dict = config.get("control_edges")
        self.eval_config: EvaluationConfig = config.get("eval_config")
        self.name_postfix: str = config.get("name_postfix")
        self.completed_reward: float = (
            self.DEF_COMPLETED_REWARD
            if config.get("completed_reward") is None
            else config.get("completed_reward")
        )
        self.results_dir = (
            self.DEF_RESULTS_DIR
            if config.get("results_dir") is None
            else config.get("results_dir")
        )

        self.sumo_config_path = (
            Path(self.sumo_config.scenario_dir) / self.sumo_config.sumo_config_file
        )

        self.network_path = get_network_file_path(self.sumo_config_path)
        # Compute time delay - initialization
        # Extract theoretical departure times from route up to simulation end time
        self.route_path = get_route_file_path(self.sumo_config_path)
        self.control_veh_type = (
            self.DEF_CONTROL_VEH_TYPE if control_veh_type is None else control_veh_type
        )

        self.random_av_switching = (
            config.get("random_av_switching")
            if config.get("random_av_switching") is not None
            else False
        )

        if self.random_av_switching:
            veh_type_counts = extract_num_veh_per_type_from_route(self.route_path)
            # and self.control_veh_type in extract_veh_types_from_route(self.route_path)
            if self.control_veh_type not in veh_type_counts.keys():
                raise ValueError(
                    f"Route file {self.route_path} must contain "
                    f"parameters for controlled vehicle type: '{self.control_veh_type}'"
                    " if random_av_switching is True."
                )

            if veh_type_counts[self.control_veh_type] > 0:
                raise ValueError(
                    f"Route file {self.route_path} must not contain vehicles of type '{self.control_veh_type}'"
                    " if random_av_switching is True."
                )

        # Setup seeding
        self.worker_index = (
            config.worker_index if hasattr(config, "worker_index") else 0
        )
        self.episode_count = 0
        self.random_av_switching_seed = (
            config.get("random_av_switching_seed")
            if config.get("random_av_switching_seed") is not None
            else None
        )

        self.av_percent = (
            config.get("av_percent") if config.get("av_percent") is not None else 0
        )
        if self.av_percent < 0 or self.av_percent > 100:
            raise ValueError(
                f"AV percentage must be between 0 and 100. Value entered: {self.av_percent}"
            )

        self.traci_conn = None
        self.episode_results_dir = self._get_episode_results_dir(self.name_postfix)
        # self.traci_conn = None
        # self.progress_bar = None
        self.highway_sorted_road_edges: List[str] = config.get(
            "highway_sorted_road_edges"
        )

        self.veh_depart_time = extract_vehicle_departure_time_from_routes(
            self.route_path, end_time=self.simulation_time
        )

        self.sim_type = "sumo-gui" if self.sumo_config.show_gui else "sumo"
        self.lateral_resolution = (
            self.sumo_config.lane_width / self.sumo_config.num_sub_lanes
        )

        self.network_road_edges = get_edge_ids(self.network_path)

        if any(
            [
                edge not in self.network_road_edges
                for edge in self.highway_sorted_road_edges
            ]
        ):
            error_edges = [
                edge
                for edge in self.highway_sorted_road_edges
                if edge not in self.network_road_edges
            ]
            raise ValueError(
                "The following edges defined in config['highway_sorted_road_edges'] "
                "are not found in the network defined by the SUMO configuration: "
                f"{error_edges}"
            )

        # initialize the simulation
        self.num_warm_up_steps = int(
            self.warm_up_time / self.sumo_config.seconds_per_step
        )
        self.num_steps = int(self.simulation_time / self.sumo_config.seconds_per_step)

        self.save_state_step_intervals = int(
            self.sumo_config.save_state_time_intervals
            / self.sumo_config.seconds_per_step
        )
        self.speed_profile_detector_file_path = (
            Path(self.sumo_config.scenario_dir)
            / self.sumo_config.speed_profile_detector_file
        )
        if not self.speed_profile_detector_file_path.exists():
            raise ValueError("Speed profile detector file does not exist.")

        self.speed_profile_detectors = get_detector_ids(
            self.speed_profile_detector_file_path
        )
        self.edge_start_locations, _, _ = edge_distance_from_start(
            self.network_path, self.highway_sorted_road_edges
        )
        self.control_edges = (
            {edge: {"start": 0, "end": -1} for edge in self.highway_sorted_road_edges}
            if self.control_edges is None
            else self.control_edges
        )
        self.edge_max_speed = get_edge_max_speed(self.network_path)
        self.normalization_speed_factor = config.get("normalization_speed_factor", 1)
        self.normalization_max_speed = (
            max(
                [
                    speed
                    for edge, speed in self.edge_max_speed.items()
                    if edge in self.control_edges.keys()
                ]
            )
            # Make sure this is higher than the actual speed of the vehicles.
            # TODO: Compute this using the speed factor of the vehicles + 3
            # sigma. Sigma is by default 0.1 according to:
            # https://sumo.dlr.de/docs/Simulation/VehicleSpeed.html#edgelane_speed_and_speedfactor
            * self.normalization_speed_factor
        )
        # TODO: Extract this from the route file?
        self.normalization_veh_len = self.DEF_VEH_LEN  # m
        self.normalization_typical_time = (
            self.normalization_veh_len / self.normalization_max_speed
        )
        self.normalization_typical_acceleration = (
            self.normalization_max_speed / self.normalization_typical_time
        )
        # The state space for each AV is its parameters and the speed/density
        # profile of the entire road. Future: use n segments behind the vehicle
        # and k segments in front of it.

        # These are the road segments for which a speed/density profile will be
        # constructed.
        self.segment_data = get_detector_segment_data(
            self.speed_profile_detector_file_path,
            self.network_path,
            self.highway_sorted_road_edges,
        )
        self.edge_segment_map = get_edge_segment_map(self.segment_data)
        self.measurement_locations = [
            segment["end"]["path_position"] for segment in self.segment_data.values()
        ]
        self.segment_lengths = np.array(
            [segment["length"] for segment in self.segment_data.values()]
        )

        self.num_road_segments = len(self.segment_data)
        # Observation space of ACC speed policy
        self.observation_space = self.get_obs_space(self.num_road_segments)

        self.edge_lanes = get_edge_num_lanes(self.network_path)
        self.num_segment_lanes = np.array(
            [
                self.edge_lanes[segment["end"]["edge"]]
                for segment in self.segment_data.values()
            ]
        )

        self.hide_libsumo_progress_bar = (
            config.get("hide_libsumo_progress_bar") or False
        )
        # self.segment_num_veh = None
        # self.segment_density = None

    # This produces an error when saving the algorithm checkpoints. RLlib
    # expects action_space to be an instance of gym Spaces, but instead it is a
    # property object.
    # @property
    # def action_space(self):
    #     # Vehicle acceleration. Both positive and negative accelerations are allowed.
    #     return spaces.Box(-float("inf"), float("inf"), shape=(1,), dtype=np.float32)

    @staticmethod
    def get_obs_space(num_road_segments):
        return spaces.Dict(
            {
                "tse_pos": spaces.Box(
                    float(0),
                    float("inf"),
                    shape=(num_road_segments,),
                    dtype=np.float32,
                ),
                "tse_speed": spaces.Box(
                    float(0),
                    float("inf"),
                    shape=(num_road_segments,),
                    dtype=np.float32,
                ),
                "ego_pos": spaces.Box(
                    float(0), float("inf"), shape=(1,), dtype=np.float32
                ),
                # May not be needed in this stage, but only for car following.
                "ego_speed": spaces.Box(
                    float(0), float("inf"), shape=(1,), dtype=np.float32
                ),
                "leader_speed": spaces.Box(
                    float(0), float("inf"), shape=(1,), dtype=np.float32
                ),
                "leader_gap": spaces.Box(
                    -float("inf"), float("inf"), shape=(1,), dtype=np.float32
                ),
            }
        )

    def observation_space_contains(self, x: MultiAgentDict) -> bool:
        for agent_obs in x.values():
            if not self.observation_space.contains(agent_obs):
                return False
        return True

    def action_space_contains(self, x: MultiAgentDict) -> bool:
        for agent_act in x.values():
            if not self.action_space.contains(agent_act):
                return False
        return True

    def observation_space_sample(self, agent_ids: list = None):
        if agent_ids is None:
            agent_ids = self.get_agent_ids()
        samples = {agent_id: self.observation_space.sample() for agent_id in agent_ids}
        return samples

    def action_space_sample(self, agent_ids: list = None):
        if agent_ids is None:
            agent_ids = self.get_agent_ids()
        return {
            agent_id: self.action_space.sample()
            for agent_id in agent_ids
            if agent_id != "__all__"
        }

    def reset(self, *args, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = {}, {}

        # Initialize parameters
        self._agent_ids = set()
        self._leader_subscribed_avs = set()
        self.segment_num_veh = None
        self.segment_density = None
        self.progress_bar = None

        self.completed_veh_ids = set()
        self.num_completed_veh = 0

        start_computation_time = time.perf_counter()
        start_process_time = time.process_time()

        # Initialize simulation
        try:
            self.close()
        except FatalTraCIError:
            pass
        except AttributeError:
            pass

        self.sumo_start_cmd = [
            self.sim_type,
            "--step-length",
            str(self.sumo_config.seconds_per_step),
            "--lateral-resolution",
            str(self.lateral_resolution),
            # "--eager-insert",
            "--extrapolate-departpos",
            "-c",
            str(self.sumo_config_path),
            "-S",  # start simulation automatically
            "--quit-on-end",  # Closes sumo gui window
            "--collision.action",  # How to resolve collisions
            "warn",  # Do nothing on collisions (no teleport or vehicle deletion)
        ]
        if self.sumo_config.no_warnings is True:
            self.sumo_start_cmd.append("--no-warnings")

        # Create unique random seed for each worker and episode
        # make it different for each rollout worker during training. See:
        # https://discuss.ray.io/t/reproducible-training-setting-seeds-for-all-workers-environments/1051/9
        worker_episode_seed = self.worker_index * 100_000 + self.episode_count
        self.episode_count += 1

        if self.sumo_config.seed is None:
            print("Sampling random seed (SUMO sim)")
            sumo_random_seed = np.random.randint(0, 1e6)
        else:
            sumo_random_seed = (self.sumo_config.seed + worker_episode_seed) % (2**32)

        self.sumo_start_cmd += ["--seed", str(sumo_random_seed)]
        # self.sumo_start_cmd += ["--log", str("log.txt")]

        self.vehicles_to_switch = set()
        if self.random_av_switching and self.av_percent > 0:
            veh_ids = extract_vehicle_ids_from_routes(self.route_path)
            num_veh_to_switch = int(len(veh_ids) * self.av_percent / 100.0)
            random_av_switching_seed = self.random_av_switching_seed
            if random_av_switching_seed is not None:
                random_av_switching_seed = (
                    random_av_switching_seed + worker_episode_seed
                ) % (2**32)

            default_rng = np.random.default_rng(seed=random_av_switching_seed)
            self.vehicles_to_switch = set(
                default_rng.choice(veh_ids, num_veh_to_switch, replace=False)
            )
            # if self.random_av_switching_seed is not None:
            #     random.seed(self.random_av_switching_seed)
            # self.vehicles_to_switch = set(random.sample(veh_ids, num_veh_to_switch))

        if self.sumo_config.use_libsumo:
            import libsumo as traci

            traci.start(self.sumo_start_cmd)
            self.traci_conn = traci
            if not self.hide_libsumo_progress_bar:
                self.progress_bar = tqdm(desc="SUMO episode", total=self.num_steps)
        else:
            import traci

            traci_conn_label = uuid.uuid4().hex
            traci.start(self.sumo_start_cmd, label=traci_conn_label)
            self.traci_conn: traci = traci.getConnection(traci_conn_label)

        if self.sumo_config.load_state_file is not None:
            if not Path(self.sumo_config.load_state_file).exists():
                raise ValueError(
                    f"Specified state file {self.sumo_config.load_state_file} does not exist"
                )

            self.traci_conn.simulation.loadState(self.sumo_config.load_state_file)

        sim_time = self.traci_conn.simulation.getTime()

        if sim_time > 0:
            self.traci_conn.simulationStep(sim_time)

        self.step_count = int(sim_time * (1 / self.sumo_config.seconds_per_step))

        # %% Subscribe to detectors and vehicles
        for detector in self.speed_profile_detectors:
            self.traci_conn.inductionloop.subscribe(
                detector,
                [
                    tc.VAR_POSITION,
                    tc.VAR_LANE_ID,
                    tc.VAR_INTERVAL_NUMBER,
                    tc.VAR_INTERVAL_SPEED,
                    tc.VAR_LAST_INTERVAL_NUMBER,
                    tc.VAR_LAST_INTERVAL_SPEED,
                ],
            )

        self._subscribe_to_veh()
        self._create_results_dir()

        veh_data = self._get_veh_data()
        _, controlled_av_data = self._get_av_data()
        self.prev_step_veh_ids = set()
        self.current_veh_ids = set(veh_data.keys())

        for veh_id in self.current_veh_ids:
            if veh_id in self.vehicles_to_switch:
                self.traci_conn.vehicle.setType(veh_id, self.control_veh_type)

        self._agent_ids = set(controlled_av_data.keys())
        if self.sumo_config.use_vehicle_based_speed_profile_update:
            self._update_speed_profile(veh_data)
        else:
            self._update_speed_profile()

        ########################### Save state ##############################
        if (
            self.sumo_config.save_states is True
            and self.step_count % self.save_state_step_intervals == 0
        ):
            self._create_states_dir(self.name_postfix)
            self._save_state()
        ########################### Evaluation ##############################
        self._init_eval()
        if self.eval_config is not None:
            self.step_computation_time = time.perf_counter() - start_computation_time
            self.step_process_time = time.process_time() - start_process_time
            self._update_metrics()
            # %% Save metadata
            self._save_metadata()
        #####################################################################

        # Update leader subscription of AVs
        self._update_leader_subscription(self._agent_ids)
        obs = self._get_initial_obs(controlled_av_data, veh_data)

        return obs, info

    def step(self, action_dict: Dict[str, Any]):
        obs, rewards, terminated, truncated, info = {}, {}, {}, {}, {}
        # Truncated: If time limit was reached or if an agent went out of bounds.
        # Terminated: If the terminal state was reached.
        terminated["__all__"] = False
        truncated["__all__"] = False
        # Start timers
        start_computation_time = time.perf_counter()
        start_process_time = time.process_time()
        # Perform actions - assumed normalized accelerations
        for av_id, av_acceleration in action_dict.items():
            self.traci_conn.vehicle.setAcceleration(
                av_id,
                float(av_acceleration * self.normalization_typical_acceleration),
                self.sumo_config.seconds_per_step,
            )

        self.traci_conn.simulationStep()  # execute one step of the simulation

        # step += 1
        sim_time = self.traci_conn.simulation.getTime()
        self.step_count = int(sim_time / self.sumo_config.seconds_per_step)
        if self.progress_bar is not None:
            self.progress_bar.update()

        if self.step_count >= self.num_steps:
            truncated["__all__"] = True
        else:
            veh_data = self._get_veh_data()
            self.prev_step_veh_ids = self.current_veh_ids
            self.current_veh_ids = set(veh_data.keys())

            # Switch vehicle type for newly entered AVs
            this_step_new_veh_ids = set(self.current_veh_ids) - set(
                self.prev_step_veh_ids
            )
            for veh_id in this_step_new_veh_ids:
                if veh_id in self.vehicles_to_switch:
                    self.traci_conn.vehicle.setType(veh_id, self.control_veh_type)

            # Update complete vehicles
            this_step_completed_veh_ids = set(self.prev_step_veh_ids) - set(
                self.current_veh_ids
            )
            self.completed_veh_ids.update(this_step_completed_veh_ids)
            self.num_completed_veh += len(this_step_completed_veh_ids)

            # Update speed profile
            if self.sumo_config.use_vehicle_based_speed_profile_update:
                self._update_speed_profile(veh_data)
            else:
                self._update_speed_profile()

            av_data, controlled_av_data = self._get_av_data()
            for av_id in av_data.keys():
                # Revert back to default speed if vehicles are not in controlled edges
                if av_id not in controlled_av_data.keys():
                    self.traci_conn.vehicle.setSpeed(av_id, -1)

            # Update leader subscription of AVs
            self._update_leader_subscription(av_data.keys())
            # Get new observations and rewards
            obs.update(self._get_obs(controlled_av_data, veh_data))
            rewards.update(self._get_speed_rewards(controlled_av_data))

            # Completed AVs
            new_agent_ids = set(controlled_av_data.keys())
            completed_agents = [
                agent_id
                for agent_id in self._agent_ids
                if agent_id not in new_agent_ids
            ]
            rewards.update(
                {agent_id: self.completed_reward for agent_id in completed_agents}
            )
            terminated.update({agent_id: True for agent_id in completed_agents})
            self._agent_ids = new_agent_ids

            ########################### Evaluation ##############################
            self.step_computation_time = time.perf_counter() - start_computation_time
            self.step_process_time = time.process_time() - start_process_time
            if self.eval_config is not None:
                self._update_metrics()
                self._update_veh_travel_info(veh_data)
            ########################### Save state ##############################
            if (
                self.sumo_config.save_states is True
                and self.step_count % self.save_state_step_intervals == 0
            ):
                self._save_state()
            #####################################################################

        return obs, rewards, terminated, truncated, info

    def close(self):
        if self.traci_conn is not None:
            if self.sumo_config.use_libsumo:
                self.traci_conn.close()
            else:
                self.traci_conn.close(False)

    def log_episode(self, episode=None):
        # TODO: Consider using mlflow for logging and saving artifacts
        if self.eval_config is not None:
            results_dir = Path(self.episode_results_dir)
            if episode is not None:
                results_dir = results_dir / f"ep_{episode}"
            results_dir.mkdir(exist_ok=True, parents=True)
            loop_detector_veh_count = step_count_from_interval_count(
                self.loop_detector_current_interval_veh_count,
                self.eval_config.throughput_loop_detector_interval_duration,
            )
            detector_result_files = get_detector_results_files(self.sumo_config_path)
            for detector_file in detector_result_files:
                shutil.copy2(detector_file, results_dir)
            # Save dataframes to csv files
            self.edge_num_veh.to_csv(Path(results_dir) / "segment_counter.csv")
            self.edge_avg_speed.to_csv(Path(results_dir) / "average_speed.csv")
            loop_detector_veh_count.to_csv(results_dir / "loop_detector_veh_count.csv")
            self.loop_detector_current_interval_veh_count.to_csv(
                results_dir / "loop_detector_current_interval_veh_count.csv"
            )
            self.step_computation_time_df.to_csv(
                results_dir / "step_computation_time.csv"
            )
            self.step_process_time_df.to_csv(results_dir / "step_process_time.csv")
            self.veh_travel_info.to_csv(results_dir / "veh_travel_info.csv")
            with open(
                str(self.episode_results_dir / "segment_data.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(self.segment_data, f, ensure_ascii=False, indent=4)

            if self.eval_config.save_lane_flows:
                self.lane_detector_flow_counts.to_csv(
                    results_dir / "lane_detector_flow_counts.csv"
                )
            if self.eval_config.save_throughput_for_speed_profile_detectors:
                speed_profile_detector_step_count = step_count_from_interval_count(
                    self.speed_profile_detector_current_interval_counts,
                    self.sumo_config.speed_profile_detector_interval_duration,
                )
                self.speed_profile_detector_current_interval_counts.to_csv(
                    results_dir / "speed_profile_detector_current_interval_counts.csv"
                )
                speed_profile_detector_step_count.to_csv(
                    results_dir / "speed_profile_detector_step_count.csv"
                )
                speed_profile_throughput_window, _ = compute_throughput(
                    speed_profile_detector_step_count,
                    self.eval_config.throughput_computation_period_time,
                    self.sumo_config.seconds_per_step,
                    int(self.warm_up_time),
                    detector_map=self.detector_group_map,
                )
                speed_profile_throughput_window.to_csv(
                    results_dir
                    / f"speed_profile_throughput_{self.eval_config.throughput_computation_period_time}_second_window.csv"
                )
            if self.eval_config.save_segment_data:
                self.segment_avg_speed_df.to_csv(results_dir / "segment_avg_speed.csv")
                self.segment_num_veh_df.to_csv(results_dir / "segment_num_veh.csv")
                self.segment_density_df.to_csv(results_dir / "segment_density.csv")

    def _get_initial_obs(self, controlled_av_data, veh_data):
        return self._get_obs(controlled_av_data, veh_data)

    def _get_veh_data(self, veh_type: str = None):
        if self.step_count == 0:
            return {}
        return get_vehicle_data(
            context_subscription_junction_id=self.veh_context_subscription_junction_id,
            veh_type=veh_type,
            traci_conn=self.traci_conn,
        )

    def _get_av_data(self):
        if self.step_count == 0:
            return {}, {}
        av_data = self._get_veh_data(self.control_veh_type)
        # TODO: Move this outside of the environment. This should return all AV data
        controlled_av_data = {
            av_id: av_params
            for av_id, av_params in av_data.items()
            if (
                (av_params[tc.VAR_ROAD_ID] in self.control_edges.keys())
                and (
                    av_params[tc.VAR_LANEPOSITION]
                    >= self.control_edges[av_params[tc.VAR_ROAD_ID]]["start"]
                )
                and (
                    (
                        av_params[tc.VAR_LANEPOSITION]
                        <= self.control_edges[av_params[tc.VAR_ROAD_ID]]["end"]
                    )
                    or self.control_edges[av_params[tc.VAR_ROAD_ID]]["end"] == -1
                )
            )
        }
        return av_data, controlled_av_data

    def _update_leader_subscription(self, av_ids):
        unneeded_subscriptions = [
            subscribed_id
            for subscribed_id in self._leader_subscribed_avs
            if subscribed_id not in av_ids
        ]

        for subscribed_id in unneeded_subscriptions:
            self._leader_subscribed_avs.remove(subscribed_id)

        new_ids = [
            av_id for av_id in av_ids if av_id not in self._leader_subscribed_avs
        ]

        # The dist parameter defines the maximum lookahead for the leader,
        # 0 calculates a lookahead from the brake gap.
        dist = self.DEF_MAX_LEADER_LOOKAHEAD
        for av_id in new_ids:
            self.traci_conn.vehicle.subscribeLeader(av_id, dist=dist)
            self._leader_subscribed_avs.add(av_id)

    def _update_speed_profile(self, veh_data=None):
        if veh_data is not None:
            # Use vehicle data to update speed profile.
            # Can be updated every time step.
            self.speed_measurements = list(
                get_segment_avg_speed(
                    veh_data,
                    self.edge_segment_map,
                    self.edge_max_speed,
                    self.segment_data,
                ).values()
            )
            self.segment_num_veh = np.array(
                list(
                    get_segment_num_vehicles(
                        veh_data, self.edge_segment_map, self.segment_data
                    ).values()
                )
            )
            self.segment_density = (
                self.segment_num_veh / self.segment_lengths / self.num_segment_lanes
            )
        else:
            # Use segment-end loop detectors to update speed profile. Updates
            # only after a time period of
            # speed_profile_detector_interval_duration seconds.
            if (
                self.step_count
                % (
                    int(
                        self.sumo_config.speed_profile_detector_interval_duration
                        / self.sumo_config.seconds_per_step
                    )
                )
                == 1
            ):
                self.measurement_locations, self.speed_measurements = get_speed_profile(
                    self.speed_profile_detectors,
                    self.edge_start_locations,
                    self.edge_max_speed,
                    traci_conn=self.traci_conn,
                )

    def _get_obs(self, controlled_av_data: dict, veh_data: dict):
        obs = {}
        subscription_results = self.traci_conn.vehicle.getAllSubscriptionResults()
        if len(controlled_av_data) > 0 and self.step_count > 0:
            av_locations = object_path_location(
                controlled_av_data, self.edge_start_locations
            )
            for av_id, av_data in controlled_av_data.items():
                ego_speed = veh_data[av_id][tc.VAR_SPEED]
                leader_speed = ego_speed
                leader_gap = self.DEF_MAX_LEADER_LOOKAHEAD
                leader_data = subscription_results[av_id][tc.VAR_LEADER]

                if leader_data is not None and not leader_data[0] == "":
                    # The SUMO 'leader' value retrieved for a vehicle gives both
                    # the leader id and the distance to the leader, measured
                    # from the front bumper + minGap to the back bumper of the
                    # leader vehicle. See:
                    # https://sumo.dlr.de/docs/TraCI/Vehicle_Value_Retrieval.html
                    leader_id, leader_gap = subscription_results[av_id][tc.VAR_LEADER]
                    leader_speed = veh_data[leader_id][tc.VAR_SPEED]

                obs[av_id] = {
                    "tse_pos": np.array(self.measurement_locations)
                    / self.normalization_veh_len,
                    "tse_speed": np.array(self.speed_measurements)
                    / self.normalization_max_speed,
                    "ego_pos": np.array([av_locations[av_id]])
                    / self.normalization_veh_len,
                    "ego_speed": np.array([ego_speed]) / self.normalization_max_speed,
                    "leader_speed": np.array([leader_speed])
                    / self.normalization_max_speed,
                    "leader_gap": np.array([leader_gap]) / self.normalization_veh_len,
                }

        return obs

    def _get_speed_rewards(self, controlled_av_data: dict):
        rewards = {}
        w_selfish = 0.5
        w_cooperative = 0.5
        # max_speed = 40  # m/s
        max_speed = self.normalization_max_speed
        avg_speed = (
            self.segment_lengths.sum()
            # Assume piecewise constant speed
            / (
                self.segment_lengths
                / np.maximum(np.array(self.speed_measurements), self.EPS)
            ).sum()
        )
        for av_id, av_params in controlled_av_data.items():
            rewards[av_id] = (
                w_cooperative * avg_speed / max_speed - w_selfish
            ) * self.sumo_config.seconds_per_step

        return rewards

    def _update_metrics(self):
        self.loop_detector_current_interval_veh_count.iloc[self.step_count] = (
            get_interval_counts(
                loop_detector_ids=self.loop_detector_current_interval_veh_count.columns,
                traci_conn=self.traci_conn,
            )
        )
        (
            self.edge_num_veh.iloc[self.step_count],
            self.edge_avg_speed.iloc[self.step_count],
        ) = get_edge_veh_count_and_average_speed(
            edge_ids=self.edge_avg_speed.columns,
            traci_conn=self.traci_conn,
        )

        self.step_computation_time_df.iloc[self.step_count] = self.step_computation_time
        self.step_process_time_df.iloc[self.step_count] = self.step_process_time

        if self.eval_config.save_lane_flows:
            (self.lane_detector_flow_counts.iloc[self.step_count]) = get_lane_flows(
                passed_veh_ids=self.passed_veh_ids,
                lane_ids=self.lane_ids,
                traci_conn=self.traci_conn,
            )

        if self.eval_config.save_throughput_for_speed_profile_detectors:
            self.speed_profile_detector_current_interval_counts.iloc[
                self.step_count
            ] = get_interval_counts(
                loop_detector_ids=self.speed_profile_detector_current_interval_counts.columns,
                traci_conn=self.traci_conn,
            )

        if self.eval_config.save_segment_data:
            self.segment_avg_speed_df.iloc[self.step_count] = self.speed_measurements
            self.segment_num_veh_df.iloc[self.step_count] = self.segment_num_veh
            self.segment_density_df.iloc[self.step_count] = self.segment_density

    def _update_veh_travel_info(self, veh_data: dict):
        current_time = self.step_count * self.sumo_config.seconds_per_step
        # Update total time and time delay
        self.veh_travel_info.loc[
            ~self.veh_travel_info.index.isin(self.completed_veh_ids)
            & (self.veh_travel_info["depart_time"] <= current_time),
            ["total_time", "time_delay"],
        ] += self.sumo_config.seconds_per_step

        veh_step_distances = self.sumo_config.seconds_per_step * np.array(
            [params[tc.VAR_SPEED] for params in veh_data.values()]
        )

        self.veh_travel_info.loc[veh_data.keys(), "time_delay"] -= (
            veh_step_distances
            / np.array(
                [
                    self.edge_max_speed[params[tc.VAR_ROAD_ID]]
                    for params in veh_data.values()
                ]
            )
        )

        self.veh_travel_info.loc[veh_data.keys(), "travel_distance"] += (
            veh_step_distances
        )

    def _save_state(self):
        self.episode_state_dir.mkdir(exist_ok=True, parents=True)
        file_name = (
            f"sim_state_{int(self.step_count * self.sumo_config.seconds_per_step)}"
            + ".xml.gz"
        )
        self.traci_conn.simulation.saveState(str(self.episode_state_dir / file_name))

    def _init_eval(self):
        if self.eval_config is not None:
            self.throughput_detector_file_path = (
                Path(self.sumo_config.scenario_dir)
                / self.eval_config.throughput_detector_file
            )
            if not self.throughput_detector_file_path.exists():
                raise ValueError(
                    f"Throughput detector file: {self.throughput_detector_file_path} does not exist."
                )

            self._initialize_metrics_data_structures()
            self._initialize_veh_travel_info()
            self._subscribe_for_metrics()

    def _get_episode_results_dir(self, name_postfix: str = None):
        return get_episode_results_dir(
            self.results_dir,
            self.sumo_config.sumo_config_file,
            self.sumo_config.scenario_dir,
            self.random_av_switching,
            self.av_percent,
            name_postfix,
        )
        # dir_name = f"{os.path.splitext(os.path.basename(self.sumo_config.sumo_config_file))[0]}"

        # if self.random_av_switching and self.av_percent > 0:
        #     dir_name += f"_random_switch_av_percent_{self.av_percent}"

        # if name_postfix is not None:
        #     dir_name += f"_{name_postfix}"

        # episode_results_dir = (
        #     Path(self.results_dir) / Path(self.sumo_config.scenario_dir).name / dir_name
        # )
        # return episode_results_dir

    def _create_results_dir(self):
        self.episode_results_dir.mkdir(exist_ok=True, parents=True)

    def _create_states_dir(self, name_postfix: str = None):
        dir_name = f"{os.path.splitext(os.path.basename(self.sumo_config.sumo_config_file))[0]}"

        if name_postfix is not None:
            dir_name += f"_{name_postfix}"

        self.episode_state_dir = (
            Path(self.sumo_config.state_dir)
            / Path(self.sumo_config.scenario_dir).name
            / dir_name
        )
        self.episode_state_dir.mkdir(exist_ok=True, parents=True)

    def _initialize_metrics_data_structures(self):
        self.edge_num_veh = pd.DataFrame(
            np.zeros((self.num_steps, len(self.highway_sorted_road_edges))),
            index=np.arange(self.num_steps) * self.sumo_config.seconds_per_step,
            columns=self.highway_sorted_road_edges,
        )

        self.main_road_west_segments_max_speed = [
            self.edge_max_speed[edge] for edge in self.highway_sorted_road_edges
        ]
        self.edge_avg_speed = pd.DataFrame(
            np.tile(self.main_road_west_segments_max_speed, (self.num_steps, 1)),
            index=np.arange(self.num_steps) * self.sumo_config.seconds_per_step,
            columns=self.highway_sorted_road_edges,
        )

        self.throughput_loop_detectors = get_detector_ids(
            self.throughput_detector_file_path
        )

        self.loop_detector_current_interval_veh_count = pd.DataFrame(
            np.zeros((self.num_steps, len(self.throughput_loop_detectors))),
            index=np.arange(self.num_steps) * self.sumo_config.seconds_per_step,
            columns=self.throughput_loop_detectors,
        )

        self.step_computation_time_df = pd.Series(
            np.zeros((self.num_steps,)),
            index=np.arange(self.num_steps) * self.sumo_config.seconds_per_step,
        )
        self.step_process_time_df = pd.Series(
            np.zeros((self.num_steps,)),
            index=np.arange(self.num_steps) * self.sumo_config.seconds_per_step,
        )

        # %% If specified, create data structure for lane detector results
        if self.eval_config is not None:
            if self.eval_config.save_lane_flows:
                lane_ids: List[str] = self.traci_conn.lane.getIDList()
                self.lane_ids = [lane for lane in lane_ids if not lane.startswith(":")]

                self.lane_detector_flow_counts = pd.DataFrame(
                    index=np.arange(self.num_steps) * self.sumo_config.seconds_per_step,
                    columns=lane_ids,
                )

                self.passed_veh_ids = {
                    loop_detector_id: set() for loop_detector_id in lane_ids
                }

            if self.eval_config.save_throughput_for_speed_profile_detectors:
                self.detector_group_map = detector_group_mapping(
                    self.speed_profile_detector_file_path,
                    sorted_road_edges=self.highway_sorted_road_edges,
                )
                self.speed_profile_detector_current_interval_counts = pd.DataFrame(
                    np.zeros((self.num_steps, len(self.detector_group_map))),
                    index=np.arange(self.num_steps) * self.sumo_config.seconds_per_step,
                    columns=self.detector_group_map.keys(),
                )

            if self.eval_config.save_segment_data:
                self.segment_num_veh_df = pd.DataFrame(
                    np.zeros((self.num_steps, len(self.segment_data))),
                    index=np.arange(self.num_steps) * self.sumo_config.seconds_per_step,
                    columns=self.segment_data.keys(),
                )
                self.segment_avg_speed_df = pd.DataFrame(
                    np.zeros((self.num_steps, len(self.segment_data))),
                    index=np.arange(self.num_steps) * self.sumo_config.seconds_per_step,
                    columns=self.segment_data.keys(),
                )
                self.segment_density_df = pd.DataFrame(
                    np.zeros((self.num_steps, len(self.segment_data))),
                    index=np.arange(self.num_steps) * self.sumo_config.seconds_per_step,
                    columns=self.segment_data.keys(),
                )

    def _initialize_veh_travel_info(self):
        # Initialize dictionaries for time delay, total time, and distance traveled
        self.veh_travel_info = pd.DataFrame(
            0.0,
            index=self.veh_depart_time.keys(),
            columns=["depart_time", "total_time", "time_delay", "travel_distance"],
        )
        self.veh_travel_info["depart_time"] = self.veh_depart_time.values()

    def _subscribe_for_metrics(self):
        for segment in self.highway_sorted_road_edges:
            self.traci_conn.edge.subscribe(
                segment, [tc.LAST_STEP_VEHICLE_NUMBER, tc.LAST_STEP_MEAN_SPEED]
            )

        for loop_detector in self.throughput_loop_detectors:
            self.traci_conn.inductionloop.subscribe(
                loop_detector, [tc.VAR_INTERVAL_NUMBER]
            )

        if self.eval_config.save_lane_flows:
            for loop_detector in self.lane_ids:
                self.traci_conn.inductionloop.subscribe(
                    loop_detector, [tc.VAR_INTERVAL_IDS]
                )

    def _subscribe_to_veh(self):
        self.veh_subscription_variables = [
            tc.VAR_ROAD_ID,
            tc.VAR_LANE_ID,
            tc.VAR_LANEPOSITION,
            tc.VAR_TYPE,
            tc.VAR_SPEED,
            # TODO: Add leader subscription
            # Subscribing to leader requires an additional parameter (dist).
            # This is not yet implemented for context subscriptions. Therefore,
            # we need to use subscribeLeader() for the current AVs... In the
            # future, we will be able to add:
            # tc.VAR_LEADER,
        ]
        self.veh_context_subscription_junction_id = get_junction_ids(self.network_path)[
            0
        ]
        self.subscription_radius = (
            1e6  # m. Very large number to include the entire network
        )

        # dist = self.MAX_LEADER_LOOKAHEAD
        self.traci_conn.junction.subscribeContext(
            self.veh_context_subscription_junction_id,
            tc.CMD_GET_VEHICLE_VARIABLE,
            self.subscription_radius,
            self.veh_subscription_variables,
            # Not yet supported in SUMO/Traci:
            # parameters={tc.VAR_LEADER: ("d", dist)},
        )

    def _save_metadata(self):
        eval_config_metadata = {}
        if self.eval_config is not None:
            eval_config_metadata.update(
                save_lane_flows=self.eval_config.save_lane_flows,
                save_throughput_for_speed_profile_detectors=self.eval_config.save_throughput_for_speed_profile_detectors,
            )
        sumo_config_metadata = dict(
            seconds_per_step=self.sumo_config.seconds_per_step,
            save_states=self.sumo_config.save_states,
            loaded_state_file=self.sumo_config.load_state_file,
            lane_width=self.sumo_config.lane_width,
            num_sub_lanes=self.sumo_config.num_sub_lanes,
        )
        metadata = dict(
            config_file=str(self.sumo_config_path),
            network_file=str(self.network_path),
            speed_profile_detector_file_path=str(self.speed_profile_detector_file_path),
            simulation_time=self.simulation_time,
            warm_up_time=self.warm_up_time,
            num_steps=self.num_steps,
            sim_type=self.sim_type,
            results_dir=str(self.episode_results_dir),
            state_dir=(
                str(self.episode_state_dir)
                if self.sumo_config.save_states is True
                else None
            ),
            sumo_config=sumo_config_metadata,
            sumo_start_cmd=" ".join(self.sumo_start_cmd),
            control_edges=self.control_edges,
            eval_config=eval_config_metadata,
        )

        with open(
            str(self.episode_results_dir / "metadata.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    scenario_dir = "scenarios/reduced_junctions"
    sumo_config_file = (
        "edge_flows_interval_8400"
        + "_taz_reduced"
        + "_flow_scale_0.8"
        # + "_custom"
        + "_av_10_percent"
        + "_highway_profile"
        + "_priority"
        + "_short_merge_lane"
        + ".sumocfg"
    )
    seconds_per_step = 0.5

    speed_profile_detector_file = "add_highway_profile_detectors.xml"
    if "short_merge_lane" in sumo_config_file:
        speed_profile_detector_file = "short_merge_lane_" + speed_profile_detector_file

    sumo_config = SumoConfig(
        scenario_dir=scenario_dir,
        sumo_config_file=sumo_config_file,
        seconds_per_step=seconds_per_step,
        show_gui=True,
        speed_profile_detector_file=speed_profile_detector_file,
        # no_warnings=True,
    )

    eval_config = EvaluationConfig()

    simulation_time = 100  # 8400
    warm_up_time = 100  # 1200

    scenario_start_edge = "977008893.1043"
    control_start_edge = "977008892"
    scenario_end_edge = "635078551"

    highway_edges = get_main_road_west_edges(with_internal=True)
    scenario_edges = highway_edges[
        highway_edges.index(scenario_start_edge) : highway_edges.index(
            scenario_end_edge
        )
        + 1
    ]
    # Start control at edge "977008892". This is the edge after "977008893.1043"
    control_edges_list = highway_edges[
        highway_edges.index(control_start_edge) : highway_edges.index(scenario_end_edge)
        + 1
    ]
    control_edges = {edge: {"start": 0, "end": -1} for edge in control_edges_list}

    env_config = dict(
        simulation_time=simulation_time,
        sumo_config=sumo_config,
        warm_up_time=warm_up_time,
        control_edges=control_edges,
        eval_config=eval_config,
        name_postfix="test_env",
        highway_sorted_road_edges=scenario_edges,
    )
    env = SumoEnv(env_config)

    num_episodes = 5
    for episode in range(num_episodes):
        obs, info = env.reset()
        terminated = {"__all__": False}
        truncated = {"__all__": False}
        episode_reward = 0

        while not any([terminated.get("__all__"), truncated.get("__all__")]):
            actions = {}

            for agent_id in obs.keys():
                # actions[vehicle_id] = random.random()  # Random action for simplicity
                actions[agent_id] = 10

            # Step the environment
            obs, rewards, terminated, truncated, info = env.step(actions)
            # Aggregate the rewards for all agents for the episode
            episode_reward += sum(rewards.values())

        print(f"Episode {episode + 1} reward: {episode_reward}")

        env.log_episode(episode)

    env.close()

    # Test running environments in parallel
    num_parallel_envs = 3
    current_episode = 0
    env_all: Dict[int, SumoEnv] = {}
    obs_all: Dict[int, dict] = {}
    episode_reward_all = {}
    actions_all: Dict[int, dict] = {}
    for env_id in range(num_parallel_envs):
        env_all[env_id] = SumoEnv(env_config)
        obs_all[env_id], _ = env_all[env_id].reset()
        episode_reward_all[env_id] = 0

    running_envs = set(range(num_parallel_envs))
    while len(running_envs) > 0:
        for env_id in running_envs:
            actions_all[env_id] = {}
            for agent_id in obs_all[env_id].keys():
                # actions[vehicle_id] = random.random()  # Random action for simplicity
                actions_all[env_id][agent_id] = 10

            # Step the environment
            obs, rewards, terminated, truncated, info = env_all[env_id].step(
                actions_all[env_id]
            )
            # Aggregate the rewards for all agents for the episode
            obs_all[env_id] = obs
            episode_reward_all[env_id] += sum(rewards.values())
            if any([terminated.get("__all__"), truncated.get("__all__")]):
                running_envs.remove(env_id)

    for env_id in range(num_parallel_envs):
        env_all[env_id].log_episode(env_id)
        env_all.pop(env_id).close()
