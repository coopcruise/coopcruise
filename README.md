# Highway Congestion Reduction through Reinforcement Learning Based Eulerian Headway Control

This repository contains the code (including SUMO and RLlib) used to train and evaluate the controllers in our ITSC 2026 paper:

> Yaron Veksler, Sharon Hornstein, Han Wang, Maria Laura Delle Monache, and Daniel Urieli.
> **Highway Congestion Reduction through Reinforcement Learning Based Eulerian Headway Control.**
> 2026 IEEE 29th International Conference on Intelligent Transportation Systems (ITSC), 2026. To appear.
> arXiv: [2412.02520](https://arxiv.org/abs/2412.02520).

- Project website (overview, figures, and example videos): [https://coopcruise.github.io](https://coopcruise.github.io)
- Paper: [https://arxiv.org/abs/2412.02520](https://arxiv.org/abs/2412.02520)

The proceedings version is forthcoming. Until a DOI is available, please cite the BibTeX entry below.

## Background

Highway congestion remains a pressing challenge in modern transportation. Highway flow through merge-bottlenecks can decrease significantly in high density traffic, creating stop-and-go waves that propagate upstream. Numerous lane-changes increase vehicle interactions and create additional slow-downs. Reducing road density near the bottleneck helps vehicles rearrange among the lanes with reduced negative effect on traffic. Currently deployed roadside variable speed limits affect density only indirectly, update a few times per hour, and depend on inconsistent driver compliance. In-Vehicle (Lagrangian) controllers can be reactive and compliant, but on a multi-lane road the useful action often depends on the hidden information of whether a neighboring driver will actually change lanes.

This work proposes a practical Eulerian (road-segment based) controller and uses vehicles equipped with adaptive cruise control (ACC) as the actuators. A reinforcement-learning policy reads average speed and density around the bottleneck and broadcasts one desired headway command for each controlled segment, to which every connected ACC vehicle adheres. Commands refresh every 2.5 seconds. Two command types are trained:

| Controller (paper) | Environment class | Command range |
| --- | --- | --- |
| TimeHeadway | `SumoEnvCentralizedTimeHeadway` | 1.5–6 s |
| DistanceHeadway | `SumoEnvCentralizedDistanceHeadway` | 0–30 m |
| SpeedLimit | `SumoEnvCentralizedSpeedLimit` | 0–31.29 m/s (70 mph) |

`SumoEnvCentralizedSpeedLimit` trains the dynamic RL speed-limit baseline. A separate constant-speed sweep approximates traditional roadside variable speed limits (Traditional VSL in the paper figures).

The scenario is a 2 km, four-lane road with a merging road in SUMO, with IDM car-following and about 1,000 vehicles per 500-second episode. Mainline inflow is at maximum capacity of 1,800 vehicles/hour/lane. Merging traffic starts after a 60-second warm-up and lasts 50 seconds. The reported grid covers merge inflows of about 7, 14, and 30 vehicles/minute and connected-ACC penetration of 20%, 60%, and 100%. Each configuration is evaluated on 30 random seeds.

At 100% connected-ACC penetration, the headway controllers raise average speed by up to **10.6%** relative to human-driven traffic, **6.7%** relative to traditional variable speed limits, and **3.4%** relative to RL speed-limit control. The comparison uses a boundary-aware speed: distance traveled inside the simulated road, divided by the time since the vehicle's planned entry. Waiting upstream of the entrance therefore counts as zero speed, so a controller cannot achieve seemingly good performance by holding vehicles out of the network.

Policies are trained with PPO in [RLlib](https://docs.ray.io/en/releases-2.7.0/rllib/index.html) 2.7. The connected-ACC subset is resampled every episode. Low-level safety stays with the ACC model: commands stay inside the ranges above.

## Citation

```bibtex
@inproceedings{veksler2026highway,
  title         = {Highway Congestion Reduction through Reinforcement Learning
                   Based Eulerian Headway Control},
  author        = {Veksler, Yaron and Hornstein, Sharon and Wang, Han and
                   Delle Monache, Maria Laura and Urieli, Daniel},
  booktitle     = {2026 IEEE 29th International Conference on Intelligent
                   Transportation Systems (ITSC)},
  year          = {2026},
  volume        = {},
  number        = {},
  note          = {To appear},
  eprint        = {2412.02520},
  archivePrefix = {arXiv},
  primaryClass  = {cs.MA}
}
```

## Setup

The experiments use Python 3.10, Ray 2.7, PyTorch 2.1, Gymnasium 0.28, and Eclipse SUMO 1.23.1 (`libsumo`, `traci`, and `sumolib`). Those versions are pinned in `environment_sumo_1.23.1.yml`.

1. Clone this repository and `cd` into it.
2. In `environment_sumo_1.23.1.yml`, set `PYTHONPATH` to the absolute path of this repository (the directory that contains `train_ppo_centralized.py`). On macOS and Linux the block should look like:

```yaml
variables:
  PYTHONPATH: $PYTHONPATH:/absolute/path/to/coopcruise
```

3. Create and activate the environment:

```bash
conda env create -f environment_sumo_1.23.1.yml
conda activate sumo_rl_1.23.1
```

Training uses libsumo and does not open the SUMO GUI. Evaluation can open the GUI with `--debug` when a display is available.

The reported results come from `train_ppo_centralized.py`, `evaluate_control_rl.py`, and `simulation_analysis.py`. `evaluate_control_new.py` is the simulation runner those evaluation scripts call.

## Reproducing the experiments

Run the commands below from the repository root, with `sumo_rl_1.23.1` active.

Pass every flag below. The first four override the script defaults.

| Flag | Paper value | Script default | What it sets |
| --- | --- | --- | --- |
| `--warm_up` | `60` | `200` | Seconds of mainline-only traffic before the merge starts |
| `--start_policy_after_warm_up` | set | off | Hold the default ACC command until the warm-up ends |
| `--num_remove_end_state_segments` | `2` | `0` | Drop the two farthest downstream segments from the observation |
| `--random_seed` | `0` | unset | SUMO seed and PPO seed. Evaluation still randomizes the ACC subset |
| `--num_control_seg` | `2` | `2` | Two controlled segments immediately upstream of the bottleneck |
| `--sim_time` | `500` | `500` | Episode length in seconds |
| `--num_workers` | `10` | `10` | Ray RLlib rollout workers during **training** only. Lower this if you have fewer cores |

`--merge_flow_percent` is the merge inflow as a percentage of the on-ramp demand in the scenario file. With the 50-second multi-lane merge used here, the paper inflows are:

| Paper inflow | `--merge_flow_percent` | Vehicles in the 50 s merge |
| --- | --- | --- |
| Light, about 7 veh/min | `25` | 6 |
| Medium, about 14 veh/min | `50` | 12 |
| Heavy, about 30 veh/min | `100` | 25 |

`--av_percent` is the connected-ACC penetration: `20`, `60`, or `100`.

`--inflow_percent` scales the mainline demand and leaves the merge demand unchanged. The default is `100`, about 1,800 vehicles/hour/lane, which is the demand used in the paper. Lower values are for exploration. `--inflow_percent 80` is about 1,440 vehicles/hour/lane. Evaluation reads the value from the checkpoint. When it is not 100, result directory names include `inflow_percent_<value>`.

Training runs 10,000 PPO iterations. It writes `checkpoint_best` (highest mean episode reward) and a checkpoint every 250 iterations under `--results_dir`. Full retraining of the 3 controllers × 3 inflows × 3 penetration rates is 27 runs and is compute-heavy. The examples use `./ray_results` so the checkpoints stay next to the repository.

### One scenario

The following is an example of launching a training run for time-headway control, medium merge, 100% connected ACC:

```bash
python train_ppo_centralized.py \
  --env_class SumoEnvCentralizedTimeHeadway \
  --av_percent 100 \
  --merge_flow_percent 50 \
  --warm_up 60 \
  --start_policy_after_warm_up \
  --num_control_seg 2 \
  --num_remove_end_state_segments 2 \
  --sim_time 500 \
  --random_seed 0 \
  --num_workers 10 \
  --results_dir ./ray_results
```

`<RUN_DIRECTORY>` below is the `PPO_SumoEnvCentralizedTimeHeadway_...` folder created under `./ray_results`. Evaluate with [Batch evaluation](#batch-evaluation) §1 and/or §2, or with a [single-checkpoint](#single-checkpoint-evaluation) call. Repeat training with `--env_class SumoEnvCentralizedDistanceHeadway` for distance headway, or `--env_class SumoEnvCentralizedSpeedLimit` for the RL speed-limit baseline.

### Full paper grid

```bash
COMMON=(
  --warm_up 60
  --start_policy_after_warm_up
  --num_control_seg 2
  --num_remove_end_state_segments 2
  --sim_time 500
  --random_seed 0
  --num_workers 10
  --results_dir ./ray_results
)

for ENV in SumoEnvCentralizedTimeHeadway SumoEnvCentralizedDistanceHeadway SumoEnvCentralizedSpeedLimit; do
  for AV in 20 60 100; do
    for FLOW in 25 50 100; do
      python train_ppo_centralized.py \
        --env_class "$ENV" \
        --av_percent "$AV" \
        --merge_flow_percent "$FLOW" \
        "${COMMON[@]}"
    done
  done
done
```

### Batch evaluation

This is the evaluation path used for the paper results. Both scripts walk a training directory (for example `./ray_results`), find every `checkpoint_best`, and call `evaluate_control_rl.py`. Scenario settings such as warm-up, inflow, and penetration are read from the checkpoint, so the paper flags do not need to be repeated. Run them from the repository root.

| § | Script | Use when |
| --- | --- | --- |
| **1** | `summarize_rl_results_av_seed.sh` | Evaluating learned headway or RL speed-limit policies (plus human-driven and no-merge baselines) |
| **2** | `summarize_rl_results_av_seed_const.sh` | Constant-command grid for traditional variable speed limits |

**Shared behavior (§1 and §2).** The batch scripts parallelize **checkpoints**. For each `checkpoint_best` under the training tree, the shell launches a background `evaluate_control_rl.py` and keeps at most 10 of those Python processes running at once. Each job uses `--num_workers 0`, so all `--num_tests` SUMO episodes for that checkpoint run **sequentially in one process**. That flag on `evaluate_control_rl.py` sets how many **stdlib multiprocessing** workers pull SUMO jobs from a queue (default `10` if you omit it).

Each job uses `--num_tests 30` (ACC-assignment seeds `0 … 29`) and `--random_seed 0`. The fixed seed pins SUMO and the policy RNG; the 30 replicates vary only in which vehicles are assigned connected ACC. If the parent run folder name lacks `av_<percent>`, that checkpoint is skipped with a warning because penetration is parsed from that substring. Ctrl-C stops the script and every evaluation process it started.

#### 1. Learned policy — `summarize_rl_results_av_seed.sh`

Evaluates the learned policy. Each call also records the human-driven baseline (policy off) and a no-merge reference. `--auto_results_dir` names the output from the checkpoint configuration. Add `--exploit` to evaluate the mean action.

```bash
bash summarize_rl_results_av_seed.sh ./ray_results
```

Optional arguments:

| Argument | Effect |
| --- | --- |
| `--exploit` | Evaluate the mean action. The results directory then ends in `_exploit` instead of `_explore`. |
| `--results_dir_prefix PREFIX` | Prepend `PREFIX` to every automatic results-directory name. |

```bash
bash summarize_rl_results_av_seed.sh ./ray_results --exploit --results_dir_prefix paper_
```

#### 2. Constant-command grid — `summarize_rl_results_av_seed_const.sh`

For every checkpoint it finds, repeats evaluation at normalized commands `0.0, 0.1, ..., 1.0`. `--const_control` uses that constant in place of the learned action. The value is a fraction of the speed limit for `SumoEnvCentralizedSpeedLimit`, and the corresponding fraction of the headway range for a headway checkpoint. Point this script at the speed-limit runs when reproducing traditional variable speed limits:

```bash
bash summarize_rl_results_av_seed_const.sh ./ray_results
```

#### 3. Evaluation results location and structure

`evaluate_control_rl.py` writes relative to the repository root (the working directory). With `--auto_results_dir`, the run name is built from the checkpoint (env class, merge/inflow, lane type, explore/exploit, optional const-control and prefix). Without it, `--results_dir` is used (default `test`).

That name is used in two places:

| Path | Contents |
| --- | --- |
| `scenarios/single_junction/<run_name>/` | Generated SUMO configs, routes, and network copies |
| `results/<run_name>/` | Per-episode simulation outputs |

For a medium-merge time-headway run, that is `results/SumoEnvCentralizedTimeHeadway_merge_flow_percent_50_multi_lane_explore/` (or the same path with `SumoEnvCentralizedTau` if the checkpoint was trained with the legacy name). Under it, each episode is a subdirectory named from the SUMO config stem, ACC penetration, control mode, and seed, for example:

```text
results/<run_name>/
  ..._random_switch_av_percent_100_rl_control_av_switch_seed_0/   # learned policy
  ..._random_switch_av_percent_100_no_control_av_switch_seed_0/  # human-driven baseline
  ..._no_merge_..._no_control_av_switch_seed_0/                  # no-merge reference
  ...  (repeated for each of the --num_tests seeds)
```

Each leaf folder holds `episode_result.json`, `metadata.json`, `veh_travel_info.csv`, segment/detector CSVs, and plots from the run. Different ACC penetrations for the same controller and merge inflow share one `results/<run_name>/` tree; only the leaf names change.

Run `simulation_analysis.py` afterwards. Scripts in §1 and §2 exit once the simulations finish. Penetration rates for the same controller and the same merge inflow share one results directory. The directory name encodes the environment class and the merge inflow, and the ACC penetration is recorded inside it. Evaluate all three penetration rates before analysis so the ACC-penetration axis is populated. If you passed `--exploit`, replace `_explore` with `_exploit` in the directory names below.

Compare the three learned controllers at medium merge (the penetration sweep in the paper):

```bash
python simulation_analysis.py \
  --results_dir \
    SumoEnvCentralizedTimeHeadway_merge_flow_percent_50_multi_lane_explore \
    SumoEnvCentralizedDistanceHeadway_merge_flow_percent_50_multi_lane_explore \
    SumoEnvCentralizedSpeedLimit_merge_flow_percent_50_multi_lane_explore
```

The script saves **`performance.json`** and a plot under each `results/.../Multi-lane/` directory (and an overlaid comparison under `results/<dir1>_vs_<dir2>/...` when you pass multiple `--results_dir` values). Repeat with `merge_flow_percent_25` and `merge_flow_percent_100` for the light and heavy inflows.

**What the plot shows:** For each ACC penetration on the x-axis, the y-value is the **mean relative speed change [%]** versus the **human-driven baseline** in the same results tree: merge inflow on, no controller (`..._no_control_...` leaves), matched AV-assignment seeds. Speed is boundary-aware (`travel_distance / total_time`, so queueing before the network counts as zero speed). Error bars are **95% confidence intervals** over the evaluation replicates (`--num_tests` / `av_switch_seed_*`). In **`performance.json`**, each penetration has `mean` and a single `std` field (standard error of the mean across seeds—not the full ± pair). The interval is symmetric: **mean ± 1.96 × std** (same half-width as the plot error bars). If only one penetration was evaluated, you get one such mean/`std` pair until you add the other penetration runs.

### Single-checkpoint evaluation

To evaluate one checkpoint without §1 or §2 (same outputs and directory layout as [§3](#3-evaluation-results-location-and-structure)):

```bash
python evaluate_control_rl.py \
  ./ray_results/<RUN_DIRECTORY>/checkpoint_best \
  --num_tests 30 \
  --num_workers 0 \
  --random_seed 0 \
  --auto_results_dir
```

On `evaluate_control_rl.py`, `--num_workers` is the number of **multiprocessing** workers for parallel SUMO episodes (`0` = one process, run episodes one after another). It is unrelated to Ray RLlib’s training rollout workers. Raise it above `0` to speed up a long single-checkpoint run; the paper batch path keeps it at `0` and relies on the shell script to run many checkpoints at once.

To watch a single run in the SUMO GUI:

```bash
python evaluate_control_rl.py \
  ./ray_results/<RUN_DIRECTORY>/checkpoint_best \
  --debug
```

The GUI opens zoomed out to the whole network, so zoom in to the second intersection from the end (Exit 59, the merge bottleneck) to see the vehicles. Blue vehicles are connected ACC in the controlled segment farther from the bottleneck; orange vehicles are connected ACC in the controlled segment next to it. Green vehicles are connected ACC outside those segments, and grey vehicles are human-driven.

### Traditional variable speed limits

Traditional VSL is a constant speed on the two controlled segments, applied while merging vehicles are on the on-ramp. The constant is chosen by a grid search in 10% steps of the speed limit. Prefer [Batch evaluation §2](#2-constant-command-grid--summarize_rl_results_av_seed_constsh) on a directory of `SumoEnvCentralizedSpeedLimit` checkpoints. For a manual sweep on one checkpoint:

```bash
VEL_CKPT=./ray_results/<VEL_RUN_DIRECTORY>/checkpoint_best

for NORM in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
  python evaluate_control_rl.py "$VEL_CKPT" \
    --const_control \
    --const_control_val_norm "$NORM" \
    --num_tests 30 \
    --num_workers 0 \
    --random_seed 0 \
    --auto_results_dir
done
```

`--const_control_val_norm` is the fraction of the segment speed limit. The value `0.9` is 90% of 31.29 m/s. All of the values for one checkpoint share one results directory, named like `SumoEnvCentralizedSpeedLimit_const_control_merge_flow_percent_50_multi_lane_explore`.

Point `--ref_results_dir` at the RL speed-limit directory from the previous section. That directory holds the human-driven baseline used in the comparison:

```bash
python simulation_analysis.py \
  --results_dir SumoEnvCentralizedSpeedLimit_const_control_merge_flow_percent_50_multi_lane_explore \
  --ref_results_dir SumoEnvCentralizedSpeedLimit_merge_flow_percent_50_multi_lane_explore
```

The summary keeps the constant level with the highest mean improvement at each penetration rate. Repeat the sweep for each inflow and penetration checkpoint used in the paper, then pass the const-control directory and the two headway directories together to overlay traditional VSL with the learned controllers. `--ref_results_dir` must be given once per `--results_dir` entry. Use the matching RL evaluation directory for each learned controller, and the matching `SumoEnvCentralizedSpeedLimit` evaluation directory for the const-control entry.
