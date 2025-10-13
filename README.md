# Cooperative Cruising: Reinforcement Learning based Headway Control for Highway Congestion Reduction

This repository contains the source code to reproduce the experiments in our paper "Cooperative Cruising: Reinforcement Learning based Headway Control for Highway Congestion Reduction."

If you find this repository helpful in your publications, please consider citing our paper.

## Introduction
Highway congestion remains one of the most pressing challenges in modern transportation. Connected automated vehicles (CAVs) equipped with adaptive cruise control (ACC) create new opportunities for congestion mitigation. Traditional practice relies on *Eulerian* variable speed limits (VSL), which regulate traffic through roadside signs but suffer from infrequent updates and limited driver compliance. More recent research has explored *Lagrangian* strategies that directly influence individual vehicles, but these approaches have struggled to deliver consistent improvements in realistic multi-lane highways, where unpredictable lane changes undermine vehicle-level decisions.

This paper introduces a reinforcement learning based Eulerian system that mitigates congestion by issuing frequent control commands to ACC-equipped vehicles approaching bottlenecks. Unlike traditional VSL, our system achieves reliable compliance and adapts dynamically to real-time traffic, while avoiding reliance on lane-change prediction by regulating density at the aggregate level. We evaluate three variants of our system---speed-limit control, time-headway control, and distance-headway control---in large-scale  simulations with thousands of vehicles across a range of merging flows and ACC penetration rates. Results show that 
headway-based variants improve traffic flow by up to 10.6\% over human traffic and 6.7\% over traditional VSL, while speed-limit control gives smaller gains.

To strengthen evaluation, we propose a novel metric for average speed in simulations with dynamic vehicle entry and exit, addressing a recognized flaw in simulation studies. Taken together, the system design,  grounded in deployable technologies, and the empirical findings indicate that Eulerian headway control of ACC-equipped vehicles opens a path toward practical, safe, and scalable congestion mitigation systems.

## Dependencies

To configure a python environment to run our code:
1. Clone the code in this repository.
1. Edit the environment_sumo_1.23.1.yml file:
    - Replace PATH\TO\CODE\DIR with the actual path to the cloned code directory.
1. install the conda environment using the edited environment_1.23.1.yml file.

List of main libraries used:
+ Python 3.x/numpy/scipy/pandas/matplotlib
+ Ray [RLlib](https://docs.ray.io/en/latest/rllib/index.html) 2.7: *A library with RL algorithm implementations and parallel training*
+ [PyTorch](https://pytorch.org) 2.1
+ Farma Foundation [gymnasium](https://gymnasium.farama.org/) 0.28: *An API standard for reinforcement learning environments*
+ [tqdm](https://tqdm.github.io/): *A library for smart progress bars*

## Instructions

### Running experiments

Experiments can be run the following command. To reproduce the results in our paper, use the parameters specified below.

```
python train_ppo_centralized.py
```
Optional parameters can be specified using the following flags:
+ Environment class to use (type of control): `--env_cls < ENVIRONMENT CLASS NAME>`. Default: `SumoEnvCentralizedTau` (time-headway); Paper: `SumoEnvCentralizedTau` or `SumoEnvCentralizedMinGap` (distance-headway) or `SumoEnvCentralizedVel` (speed limit).
+ ACC-equipped vehicle percentage: `--av_percent <AV_PERCENT>`. Default: `100`; Paper: `100` or `60` or `20`
+ Simulation running time before merging traffic starts: `--warm_up`. Default: `200`; Paper: `60`.
+ Merging traffic flow percent of maximum inflow: `--merge_flow_percent`. Default: `100`; Paper: `100` or `50` or `25`.
+ Use single-lane scenario: `--single_lane`. Default & paper: False
+ Number of segments before the bottleneck within which to control ACC-equipped vehicles: `--num_control_seg <NUM_CONTROL_SEGMENTS>`. Default & paper: `2`
+ Simulation time horizon: `--sim_time <SIMULATION_TIME_HORIZON>`. Default & paper: `500`
+ Random seed: `--random_seed <SEED INT>`. Default: None; Paper results: `0`
+ Flag to start using actions from policy only after warm-up is completed. `--start_policy_after_warm_up`. Default: False; Paper: True
+ Number of simulated scenario highway segments (counting from most downstream) to remove from the state. `--num_remove_end_state_segments`. Default: `0`; Paper: `2`
+ Number of parallel rollout workers: `--num_workers <NUM_WORKERS>`. Default & paper: `10`

### Evaluating controllers

After training using to code above, RL controllers can be evaluated using the following command.
```
python evaluate_control_rl.py <CHECKPOINT_DIR_PATH>
```
optional parameters:
+ Number of tests to run: `--num_tests <NUMBER_OF_TESTS>`. Default: `30`
+ Results output directory name: `--results_dir <DIRECTORY_NAME>`. Default: `test`
+ Flag to automatically create results directory name (recommended). `--auto_results_dir`. Default: False.

### Computeing aggregate average speed metric

After running the evaluation above, aggregate metrics can be computed using the following script. Note that you must specify the required results directories to use with the `--results_dir` argument.
```
python simulation_analysis.py --results_dir <RESULTS DIRECTORY 1> <RESULTS DIRECTORY 2> <RESULTS DIRECTORY 3> ...
```
Parameters:
+ Evaluation results directory name: `--results_dir <DIRECTORY_NAME>`. Default: 'test'. You may specify more than one directory to compute performance metrics for all given directories.
