import numpy as np
import pandas as pd
import traci
import traci.constants as tc
from typing import Dict, Iterable
from bisect import bisect

from utils.sumo_utils import object_path_location

USE_SUBSCRIPTIONS = True
VEH_BASED_METRICS = False
USE_AV_CONTEXT_SUBSCRIPTIONS = True


def get_interval_counts(
    loop_detector_ids: Iterable[str],
    use_subscriptions=USE_SUBSCRIPTIONS,
    traci_conn=None,
):
    """Get current vehicle counts for each required loop detector.

    Args:
        loop_detector_ids:
            A list of loop detector IDs.
        use_subscriptions:
            Whether to use subscription results to get flows. Assumes that traci
            is subscribed to all required detectors. Defaults to
            USE_SUBSCRIPTIONS.
        traci_conn:
            Traci connection to use. If None, calls traci.<function> instead.

    Returns:
        A list of vehicle counts per loop detector.
    """
    traci_conn = traci if traci_conn is None else traci_conn
    loop_detector_current_interval_counts = [
        traci_conn.inductionloop.getSubscriptionResults(loop_detector_id)[
            tc.VAR_INTERVAL_NUMBER
        ]
        if use_subscriptions
        else len(traci_conn.inductionloop.getIntervalVehicleNumber(loop_detector_id))
        for loop_detector_id in loop_detector_ids
    ]
    return loop_detector_current_interval_counts


def get_edge_veh_count_and_average_speed(
    edge_ids: Iterable[str] = None,
    veh_ids: Iterable[str] = None,
    veh_based_metrics=VEH_BASED_METRICS,
    use_subscriptions=USE_SUBSCRIPTIONS,
    traci_conn=None,
):
    """Get current vehicle count and average speed for required edges.

    Args:
        edge_ids:
            Required edge IDs for which to extract vehicle counts and average
            speed. If None, extracts all edge IDs from Traci. Defaults to None.
        veh_ids:
            Required vehicle IDs to use for vehicle counts and average speed
            computation. Only used if veh_based_metrics is True. If None,
            extracts all vehicle IDs from Traci. Defaults to None.
        veh_based_metrics:
            Whether to use vehicle-based computations. This may slow the
            simulation considerably if not using subscriptions. Defaults to
            VEH_BASED_METRICS.
        use_subscriptions:
            Whether to use subscription results to get flows. Assumes that traci
            is subscribed to all relevant edges. Defaults to USE_SUBSCRIPTIONS.
        traci_conn:
            Traci connection to use. If None, calls traci.<function> instead.

    Returns:
        A tuple of two lists:
            A list of edge vehicle counts, and a list of edge average speed.
    """
    traci_conn = traci if traci_conn is None else traci_conn
    edge_ids = traci_conn.edge.getIDList() if edge_ids is None else edge_ids
    if veh_based_metrics:
        veh_ids = traci_conn.vehicle.getIDList() if veh_ids is None else veh_ids
        vehicle_edges = pd.Series(
            {
                veh_id: traci_conn.vehicle.getRoadID(veh_id)
                for veh_id in traci_conn.vehicle.getIDList()
            }
        )
        vehicle_speed = pd.Series(
            {
                veh_id: traci_conn.vehicle.getSpeed(veh_id)
                for veh_id in traci_conn.vehicle.getIDList()
            }
        )
        edge_avg_speed = (
            pd.DataFrame({"edge": vehicle_edges, "speed": vehicle_speed})
            .groupby("edge")
            .mean()["speed"]
        )
        common_edges = edge_avg_speed.index[edge_avg_speed.index.isin(edge_ids)]
        return (
            vehicle_edges.value_counts()[common_edges],
            edge_avg_speed[common_edges],
        )

    else:
        edges = edge_ids
        edge_num_veh = [
            traci_conn.edge.getSubscriptionResults(edge)[tc.LAST_STEP_VEHICLE_NUMBER]
            if use_subscriptions
            else traci_conn.edge.getLastStepVehicleNumber(edge)
            for edge in edges
        ]

        edge_avg_speed = [
            traci_conn.edge.getSubscriptionResults(edge)[tc.LAST_STEP_MEAN_SPEED]
            if use_subscriptions
            else traci_conn.edge.getLastStepMeanSpeed(edge)
            for edge in edges
        ]

        return edge_num_veh, edge_avg_speed


def get_lane_flows(
    passed_veh_ids: Dict[str, set],
    lane_ids: Iterable[str],
    use_subscriptions=USE_SUBSCRIPTIONS,
    traci_conn=None,
):
    """Get current lane flow for required lanes. Assumes lane detector IDs are
    similar to lane IDs. This function updates the values in passed_veh_ids.

    Args:
        passed_veh_ids:
            Dictionary with lane IDs as keys and a set of vehicle IDs that went
            through the detector in previous time steps as values.
        lane_ids:
            List of lane IDs to use as detector IDs.
        use_subscriptions:
            Whether to use subscription results to get flows. Assumes that traci
            is subscribed to all detectors. Defaults to USE_SUBSCRIPTIONS.
        traci_conn:
            Traci connection to use. If None, calls traci.<function> instead

    Returns:
        A list of flow count dictionaries in string form. The length of the list
        is the number of required lanes, while each value is formatted as
        follows:
            '{"destination_id_1": num_vehicles_to_destination_id_1,
            "destination_id_2": num_vehicles_to_destination_id_2,
            "destination_id_3": num_vehicles_to_destination_id_3, ...}'

    """
    traci_conn = traci if traci_conn is None else traci_conn
    lane_loop_step_destinations = [
        {
            veh_id: traci_conn.vehicle.getRoute(veh_id)[-1]
            for veh_id in traci_conn.inductionloop.getSubscriptionResults(
                loop_detector_id
            )[tc.VAR_INTERVAL_IDS]
            if veh_id not in passed_veh_ids[loop_detector_id]
        }
        if use_subscriptions
        else {
            veh_id: traci_conn.vehicle.getRoute(veh_id)[-1]
            for veh_id in traci_conn.inductionloop.getIntervalVehicleIDs(
                loop_detector_id
            )
            if veh_id not in passed_veh_ids[loop_detector_id]
        }
        for loop_detector_id in lane_ids
    ]

    for loop_detector_id, detector_new_veh_data in zip(
        lane_ids, lane_loop_step_destinations
    ):
        for veh_id in detector_new_veh_data.keys():
            passed_veh_ids[loop_detector_id].add(veh_id)

    lane_loop_flow_counts = [
        str(
            dict(
                (dest, list(destinations.values()).count(dest))
                for dest in set(destinations.values())
            )
        ).replace("'", '"')
        for destinations in lane_loop_step_destinations
    ]
    return lane_loop_flow_counts


def get_av_data(
    subscribed_ids: set = None,
    av_ids: Iterable[str] = None,
    av_subscription_variables: list = None,
    av_context_subscription_junction_id: str = None,
    use_av_context_subscriptions=USE_AV_CONTEXT_SUBSCRIPTIONS,
    traci_conn=None,
):
    traci_conn = traci if traci_conn is None else traci_conn
    """Extract AV data from simulation.

    Args:
        subscribed_ids:
            A set of AV IDs that Traci is subscribed to. This function
            changes this set by adding and removing AV IDs. The function assumes
            that Traci is subscribed to all IDs in this list. Required only if
            not using AV context subscriptions. Defaults to None.
        av_ids:
            A list of all AV IDs in the simulation route file. Required only if
            not using AV context subscriptions. Defaults to None.
        av_subscription_variables:
            A list of variables to subscribe to when new AVs join the
            simulation. The variables must be traci constants defined in
            traci.constants. Required only if not using AV context
            subscriptions. Defaults to None.
        av_context_subscription_junction_id:
            Junction id that defines the context subscription to get vehicle
            data. Required only if using AV context subscriptions. Defaults to
            None.
        use_av_context_subscriptions:
            Whether to use context subscriptions to receive AV data. Requires
            av_context_subscription_junction_id to be defined, and assumes a
            context subscription to all vehicles was defined for this junction
            id. Defaults to USE_AV_CONTEXT_SUBSCRIPTIONS.

    Raises:
        ValueError:
            If av_context_subscription_junction_id is not defined but
            use_av_context_subscriptions is True.
        ValueError:
            If use_av_context_subscriptions is False but subscribed_ids, av_ids,
            or av_subscription_variables are not defined.

    Returns:
        A dictionary of AV IDs as keys and AV parameter dictionaries as values.
    """
    traci_conn = traci if traci_conn is None else traci_conn
    if use_av_context_subscriptions:
        if av_context_subscription_junction_id is None:
            raise ValueError(
                "av_context_subscription_junction_id must be defined if using AV context subscriptions"
            )
        return {
            veh_id: params
            for veh_id, params in traci_conn.junction.getContextSubscriptionResults(
                av_context_subscription_junction_id
            ).items()
            if params[tc.VAR_TYPE] == "AV"
        }

    if subscribed_ids is None or av_ids is None or av_subscription_variables is None:
        raise ValueError(
            "subscribed_ids, av_ids, and av_subscription_variables must be defined "
            + "if not using AV context subscriptions"
        )

    current_av_ids = [
        veh_id for veh_id in traci_conn.vehicle.getIDList() if veh_id in av_ids
    ]
    new_av_ids = [veh_id for veh_id in current_av_ids if veh_id not in subscribed_ids]
    completed_av_ids = [
        veh_id for veh_id in subscribed_ids if veh_id not in current_av_ids
    ]

    for veh_id in new_av_ids:
        traci_conn.vehicle.subscribe(veh_id, av_subscription_variables)
        subscribed_ids.add(veh_id)

    for veh_id in completed_av_ids:
        # No need to unsubscribe if the vehicle left the simulation. It does it automatically
        # traci_conn.vehicle.unsubscribe(veh_id)
        subscribed_ids.remove(veh_id)

    return {
        veh_id: veh_params
        for veh_id, veh_params in traci_conn.vehicle.getAllSubscriptionResults().items()
        if veh_id in subscribed_ids
    }


def get_vehicle_data(
    context_subscription_junction_id: str, veh_type: str = None, traci_conn=None
):
    """Extract AV data from simulation.

    Args:
        context_subscription_junction_id:
            Junction id that defines the context subscription to get vehicle
            data.
        veh_type:
            Type of vehicle. If defined, returns only the parameters of this
            vehicle type. Defaults to None.

    Returns:
        A dictionary of vehicle IDs as keys and vehicle parameter dictionaries
        as values.
    """
    traci_conn = traci if traci_conn is None else traci_conn
    return {
        veh_id: params
        for veh_id, params in traci_conn.junction.getContextSubscriptionResults(
            context_subscription_junction_id
        ).items()
        if veh_type is None
        or str(params[tc.VAR_TYPE]).lower().split("@")[0] == str(veh_type).lower()
    }


def get_speed_profile(
    speed_profile_detectors: Iterable[str],
    edge_start_locations: Dict[str, float],
    edge_max_speed: Dict[str, float],
    traci_conn=None,
):
    """Computes average speed profile of road.
    For each road location, the average speed of all lanes is averaged with
    weights equal to the number of vehicles that passed each lane. If no
    vehicles went through any of the lanes in a certain location, it is assumed
    that the average speed is the speed limit of the edge.

    Args:
        speed_profile_detectors:
            A list of speed profile detector IDs. This function assumes that
            traci is subscribed to all speed profile detectors with parameters:
            tc.VAR_LANE_ID, tc.VAR_POSITION, tc.VAR_LAST_INTERVAL_NUMBER,
            tc.VAR_LAST_INTERVAL_SPEED.
        edge_start_locations:
            An ordered dictionary with edge IDs as keys and start location as
            values. This can be computed using the edge_distance_from_start
            function using the network file and an ordered list of edges in the
            route path.
        edge_max_speed:
            A dictionary with edge IDs as keys and maximum speed as values.

    Returns:
        A tuple of:
            An ordered list of detector locations measured from the start of the
            road. An ordered list of average speed in each location.
    """
    traci_conn = traci if traci_conn is None else traci_conn
    detector_measurements = {
        detector: traci_conn.inductionloop.getSubscriptionResults(detector)
        for detector in speed_profile_detectors
    }

    for measurements in detector_measurements.values():
        edge = measurements[tc.VAR_LANE_ID].split("_")[0]
        measurements[tc.VAR_ROAD_ID] = edge

    detector_path_locations = object_path_location(
        detector_measurements, edge_start_locations, location_key=tc.VAR_POSITION
    )
    measurement_locations = sorted(
        set(
            [
                location
                for location in detector_path_locations.values()
                if location is not None
            ]
        )
    )

    detector_by_location = {}
    for detector_id, measurements in detector_measurements.items():
        detector_location = detector_path_locations[detector_id]
        if detector_location in detector_by_location.keys():
            detector_by_location[detector_path_locations[detector_id]][detector_id] = (
                measurements
            )
        else:
            detector_by_location[detector_path_locations[detector_id]] = {
                detector_id: measurements
            }

    speed_measurements = []
    for location in measurement_locations:
        location_measurements: dict = detector_by_location[location]
        num_vehicles = [
            measurements[tc.VAR_LAST_INTERVAL_NUMBER]
            for measurements in location_measurements.values()
        ]
        mean_speed = [
            measurements[tc.VAR_LAST_INTERVAL_SPEED]
            for measurements in location_measurements.values()
        ]
        if any(num_vehicles):
            speed_measurements.append(np.average(mean_speed, weights=num_vehicles))
        # If no vehicles went through any of the detectors, mean speed is -1.
        # Instead we would like to use the speed limit of the edge
        else:
            edge = list(location_measurements.values())[0][tc.VAR_ROAD_ID]
            speed_measurements.append(edge_max_speed[edge])

    return measurement_locations, speed_measurements


def get_segment_avg_speed(
    veh_data: dict, edge_segment_map: dict, edge_max_speed, segment_data: dict
):
    """Compute average speed for each road segment

    Args:
        veh_data:
            A dictionary of vehicle IDs as keys and vehicle parameter
            dictionaries as values. Vehicle parameter dictionaries must include
            the following keys:
                `tc.VAR_ROAD_ID`:
                    Vehicle edge ID
                `tc.VAR_LANEPOSITION`:
                    Vehicle longitudinal lane position (measured from edge
                    start)
                `tc.VAR_SPEED`:
                    Vehicle speed.
        edge_segment_map:
            A dictionary with edge IDs as keys and segment data as values.
            Values are dictionaries with the following keys:
                `"positions"`:
                    A list of segment end positions within the edge.
                `"idx"`:
                    A list of segment indices. These indices are global indices
                    of all road segments.
        edge_max_speed:
            A dictionary mapping edge IDs to edge maximum allowed speed.
        segment_data:
            A dictionary with segment IDs as keys and corresponding segment data
            as values. The value must at least include segment end point data
            under the `"end"` key. The end point data is a dictionary with the
            following keys:
                `"edge"`:
                    Segment end point edge ID (str).
                `"edge_position"`:
                    Segment end point position along the edge.

    Returns:
        A dictionary mapping segment ID to average speed.
    """
    segment_ids = list(segment_data.keys())

    segment_vehicle_speed = {segment_id: [] for segment_id in segment_ids}
    for vehicle_params in veh_data.values():
        edge = vehicle_params[tc.VAR_ROAD_ID]
        if edge in edge_segment_map.keys():
            segment_idx = edge_segment_map[edge]["idx"][
                bisect(
                    edge_segment_map[edge]["positions"][:-1],
                    vehicle_params[tc.VAR_LANEPOSITION],
                )
            ]
            segment_vehicle_speed[segment_ids[segment_idx]].append(
                vehicle_params[tc.VAR_SPEED]
            )

    return {
        segment_id: np.average(vehicle_speeds)
        if len(vehicle_speeds) > 0
        else edge_max_speed[segment_data[segment_id]["end"]["edge"]]
        for segment_id, vehicle_speeds in segment_vehicle_speed.items()
    }


def get_segment_num_vehicles(
    veh_data: dict, edge_segment_map: dict, segment_data: dict, veh_type: str = None
):
    """Compute number of vehicles within each road segment

    Args:
        veh_data:
            A dictionary of vehicle IDs as keys and vehicle parameter
            dictionaries as values. Vehicle parameter dictionaries must include
            the following keys:
                `tc.VAR_ROAD_ID`:
                    Vehicle edge ID
                `tc.VAR_LANEPOSITION`:
                    Vehicle longitudinal lane position (measured from edge
                    start)
                `tc.VAR_TYPE` (optional):
                    A string representing the vehicle type. Needed only if the
                    veh_type parameter is specified.
        edge_segment_map:
            A dictionary with edge IDs as keys and segment data as values.
            Values are dictionaries with the following keys:
                `"positions"`:
                    A list of segment end positions within the edge.
                `"idx"`:
                    A list of segment indices. These indices are global indices
                    of all road segments.
        segment_data:
            A dictionary with segment IDs as keys and corresponding segment data
            as values. Used to extract an ordered list of segment IDs.
        veh_type (optional):
            A vehicle type string for filtering vehicles. If specified, only
            counts vehicles of the specified type.

    Returns:
        A dictionary mapping segment ID to number of vehicles.
    """
    segment_ids = list(segment_data.keys())

    segment_num_vehicles = {segment_id: 0 for segment_id in segment_ids}
    for vehicle_params in veh_data.values():
        if (
            veh_type is not None
            and not str(vehicle_params[tc.VAR_TYPE]).lower().split("@")[0]
            == str(veh_type).lower()
        ):
            continue

        edge = vehicle_params[tc.VAR_ROAD_ID]
        if edge in edge_segment_map.keys():
            segment_idx = edge_segment_map[edge]["idx"][
                bisect(
                    # Here we assume that the last segment ends at the end of
                    # the edge. Therefore we bisect only on [:-1], since if the
                    # vehicle is between the last segment end and the end of the
                    # edge we will get an out of bounds error.
                    edge_segment_map[edge]["positions"][:-1],
                    vehicle_params[tc.VAR_LANEPOSITION],
                )
            ]
            segment_num_vehicles[segment_ids[segment_idx]] += 1

    return segment_num_vehicles


def get_segment_veh_ids(
    veh_data: dict, edge_segment_map: dict, segment_data: dict, veh_type: str = None
):
    """Extract list of vehicle IDs for each road segment.

    Args:
        veh_data:
            A dictionary of vehicle IDs as keys and vehicle parameter
            dictionaries as values. Vehicle parameter dictionaries must include
            the following keys:
                `tc.VAR_ROAD_ID`:
                    Vehicle edge ID
                `tc.VAR_LANEPOSITION`:
                    Vehicle longitudinal lane position (measured from edge
                    start)
                `tc.VAR_TYPE` (optional):
                    A string representing the vehicle type. Needed only if the
                    veh_type parameter is specified.
        edge_segment_map:
            A dictionary with edge IDs as keys and segment data as values.
            Values are dictionaries with the following keys:
                `"positions"`:
                    A list of segment end positions within the edge.
                `"idx"`:
                    A list of segment indices. These indices are global indices
                    of all road segments.
        segment_data:
            A dictionary with segment IDs as keys and corresponding segment data
            as values. Used to extract an ordered list of segment IDs.
        veh_type (optional):
            A vehicle type string for filtering vehicles. If specified, only
            lists vehicles of the specified type.

    Returns:
        A dictionary mapping segment ID to list of vehicle IDs.
    """
    segment_ids = list(segment_data.keys())

    segment_veh_ids = {segment_id: [] for segment_id in segment_ids}
    for veh_id, vehicle_params in veh_data.items():
        if (
            veh_type is not None
            and not str(vehicle_params[tc.VAR_TYPE]).lower() == str(veh_type).lower()
        ):
            continue

        edge = vehicle_params[tc.VAR_ROAD_ID]
        if edge in edge_segment_map.keys():
            segment_idx = edge_segment_map[edge]["idx"][
                bisect(
                    edge_segment_map[edge]["positions"][:-1],
                    vehicle_params[tc.VAR_LANEPOSITION],
                )
            ]
            segment_veh_ids[segment_ids[segment_idx]].append(veh_id)

    return segment_veh_ids
