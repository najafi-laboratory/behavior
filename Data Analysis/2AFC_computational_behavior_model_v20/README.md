# 2AFC Computational Behavior Model v20

This project is the modular Python version of [`Model_v20.ipynb`](/home/ihsan/Desktop/data/Georgia_Tech/2AFC/Codes/Model_behavior/2AFC_computational_behavior_model_v20/Model_v20.ipynb). It fits a 14-parameter PyTorch model to mouse 2AFC behavior, simulates trial-by-trial internal states, and exports the fitted results for downstream analysis.

The code is set up to run on Georgia Tech PACE with Slurm array jobs, but it can also be run locally on a single machine.

## What The Model Does

For each behavioral session, the model:

- Loads one MATLAB `.mat` file containing `SessionData`.
- Extracts trial outcomes, ISI, block type, and inferred mouse choice.
- Fits a 14-parameter dynamic decision model using a 3-stage Adam optimization.
- Simulates the fitted model trial-by-trial.
- Saves:
  - trial-level simulated/internal-state output
  - fitted parameters repeated per session
  - optimization loss history

The model includes:

- internal noisy time perception
- block/context belief updating
- separate reward and punishment learning
- choice history bias
- dynamic decision boundary updates
- rare-trial weighting
- soft barrier penalties near parameter bounds

## Folder Structure

- [`main.py`](/home/ihsan/Desktop/data/Georgia_Tech/2AFC/Codes/Model_behavior/2AFC_computational_behavior_model_v20/main.py): main execution entrypoint for local runs and PACE jobs
- [`Main.sh`](/home/ihsan/Desktop/data/Georgia_Tech/2AFC/Codes/Model_behavior/2AFC_computational_behavior_model_v20/Main.sh): Slurm submission script for PACE
- [`constants.py`](/home/ihsan/Desktop/data/Georgia_Tech/2AFC/Codes/Model_behavior/2AFC_computational_behavior_model_v20/constants.py): notebook-derived constants
- [`data_paths.txt`](/home/ihsan/Desktop/data/Georgia_Tech/2AFC/Codes/Model_behavior/2AFC_computational_behavior_model_v20/data_paths.txt): one `.mat` path per line for array jobs
- [`Module/data_processing.py`](/home/ihsan/Desktop/data/Georgia_Tech/2AFC/Codes/Model_behavior/2AFC_computational_behavior_model_v20/Module/data_processing.py): data loading and dataframe conversion
- [`Module/model_definition.py`](/home/ihsan/Desktop/data/Georgia_Tech/2AFC/Codes/Model_behavior/2AFC_computational_behavior_model_v20/Module/model_definition.py): core PyTorch mouse model
- [`Module/model_fitting.py`](/home/ihsan/Desktop/data/Georgia_Tech/2AFC/Codes/Model_behavior/2AFC_computational_behavior_model_v20/Module/model_fitting.py): 3-stage optimization logic
- [`Module/model_simulation.py`](/home/ihsan/Desktop/data/Georgia_Tech/2AFC/Codes/Model_behavior/2AFC_computational_behavior_model_v20/Module/model_simulation.py): JIT fitting loop and detailed simulation output
- [`Module/fitting_metrics.py`](/home/ihsan/Desktop/data/Georgia_Tech/2AFC/Codes/Model_behavior/2AFC_computational_behavior_model_v20/Module/fitting_metrics.py): barrier penalty, restart selection, tensor conversion
- [`Module/results_export.py`](/home/ihsan/Desktop/data/Georgia_Tech/2AFC/Codes/Model_behavior/2AFC_computational_behavior_model_v20/Module/results_export.py): CSV export
- [`Model_v20.ipynb`](/home/ihsan/Desktop/data/Georgia_Tech/2AFC/Codes/Model_behavior/2AFC_computational_behavior_model_v20/Model_v20.ipynb): original notebook reference

## Input Data Requirements

Each input file should be a MATLAB `.mat` file containing a `SessionData` field compatible with the loader in [`Module/Reader.py`](/home/ihsan/Desktop/data/Georgia_Tech/2AFC/Codes/Model_behavior/2AFC_computational_behavior_model_v20/Module/Reader.py).

The filename is expected to match the pattern parsed by [`Module/session_name_parse.py`](/home/ihsan/Desktop/data/Georgia_Tech/2AFC/Codes/Model_behavior/2AFC_computational_behavior_model_v20/Module/session_name_parse.py), for example:

```text
MC06_SChR2_block_single_interval_discrimination_V_1_20250801_144807.mat
```

## Output Files

For each run, the code writes two CSV files into the chosen save directory:

- `Dynamic_Boundary_Fit_TaskX_Trials_and_Params.csv`
- `Dynamic_Boundary_Fit_TaskX_Loss_History.csv`

`TaskX` is based on `SLURM_ARRAY_TASK_ID`.

The trial output CSV contains:

- trial metadata
- model choice
- model correctness
- pre-trial internal states
- fitted parameters
- final NLL
- fixed bias parameter

## Trial Output Column Guide

The `*_Trials_and_Params.csv` file contains trial-by-trial traces plus the fitted parameters copied onto every row for that session.

### Metadata and observed behavior

- `session_id`: internal session index used by the pipeline
- `trial_in_session`: zero-based trial number within the session
- `isi`: objective ISI for that trial
- `trial_type`: `short` or `long`
- `block_type`: `neutral`, `short_block`, or `long_block`
- `mouse_choice`: inferred mouse choice from the real data
- `model_choice`: simulated model choice
- `correct_model`: whether the model’s simulated choice was correct
- `is_rare_trial`: whether the trial type is rare under the current block belief
- `trials_since_switch`: number of trials since the current block started

### Pre-trial internal state

- `p_short_pre`: belief that the current block is a short block before the trial
- `w_time_pre`: sensory branch weight before update
- `w_ctx_pre`: context branch weight before update
- `bias_pre`: choice-history bias before update
- `boundary_pre`: physical decision boundary before update
- `fixed_bias`: static neutral-block bias used during simulation
- `lapse_value`: fitted lapse parameter used on that trial

### Boundary mapping and time perception

- `boundary_cdf_pre`: CDF-mapped boundary value before normalization
- `internal_boundary`: internal boundary after mapping from physical boundary
- `mapped_time_cdf`: CDF-mapped value of the trial ISI
- `mapped_time_mean`: mean mapped internal time before sensory noise
- `mapped_time_std`: standard deviation used for sensory noise
- `added_noise_mean`: average added sensory noise for the sampled internal times
- `added_noise_std`: spread of the sampled sensory noise
- `perceived_time_mean`: mean noisy internal time used for learning

### Sensory branch calculations

- `sensory_dist_mean`: mean perceived time minus internal boundary
- `sensory_exp_arg`: exponent argument before the sensory sigmoid
- `sensory_sigmoid`: sigmoid output for the sensory branch
- `sensory_evidence`: transformed sensory evidence term `2*sigmoid-1`
- `dv_time_mean`: sensory contribution to the decision variable after multiplying by sensory weight
- `time_is_perceived_short`: whether the perceived time falls below the internal boundary
- `time_correct_direction`: whether the sensory branch points in the correct task direction
- `dir_time`: signed direction used for the sensory weight update

### Context branch calculations

- `context_evidence_pre`: context evidence before update, computed as `0.5 - p_short_pre`
- `dv_context`: context contribution to the decision variable after multiplying by context weight
- `uncertainty_pre`: uncertainty term `4 * p_short_pre * (1 - p_short_pre)`
- `p_short_prior`: prior short-block belief before the Bayesian update
- `likelihood_short`: likelihood of the observed trial under the short block hypothesis
- `likelihood_long`: likelihood of the observed trial under the long block hypothesis
- `bayes_numerator`: numerator of the Bayesian context update
- `bayes_denominator`: denominator of the Bayesian context update
- `p_short_post`: updated short-block belief after the trial
- `dir_context`: signed direction used for the context weight update

### Decision calculations

- `decision_variable_mean`: average full decision variable across noisy sensory samples
- `decision_variable_no_lapse_mean`: decision variable built from the mean sensory term before lapse
- `decision_beta_scaled_mean`: beta-scaled decision variable before the final sigmoid
- `decision_sigmoid_mean`: final decision sigmoid output before lapse
- `decision_without_lapse`: left-choice probability before lapse is applied
- `decision_with_lapse`: final left-choice probability after lapse

### Boundary and weight updates

- `boundary_shift`: signed boundary adjustment caused by an incorrect trial
- `boundary_post`: physical boundary after update
- `alpha_base`: base learning rate chosen from reward or punishment
- `alpha_context`: effective learning rate for the context branch
- `alpha_sensory`: effective learning rate for the sensory branch
- `delta_w_time_raw`: raw sensory weight change before non-negativity clamp
- `delta_w_ctx_raw`: raw context weight change before non-negativity clamp
- `delta_w_time_applied`: actual sensory weight change after clamping
- `delta_w_ctx_applied`: actual context weight change after clamping
- `w_time_post`: sensory weight after update
- `w_ctx_post`: context weight after update

### Bias history update

- `history_alpha`: learning rate used for choice-history bias update on that trial
- `target_bias`: bias target implied by choice and outcome
- `bias_delta`: change in choice-history bias on that trial
- `bias_post`: updated choice-history bias after the trial

### Fitted parameters repeated on every row

- `decay_rate`
- `noise_param_a`
- `alpha_reward`
- `alpha_punish`
- `gamma`
- `alpha_unc_ctx`
- `alpha_unc_sens`
- `alpha_ch_r`
- `alpha_ch_p`
- `beta`
- `lapse`
- `p_switch`
- `p_rare`
- `alpha_boundary`
- `fixed_bias_param`
- `initial_boundary_param`
- `p_common_param`
- `final_nll`

## Python Dependencies

Make sure your environment includes:

- `python`
- `numpy`
- `pandas`
- `scipy`
- `torch`
- `tqdm`

Example local install:

```bash
pip install numpy pandas scipy torch tqdm
```

## Running Locally

### Option 1: Single file

```bash
python main.py \
  --data-path /absolute/path/to/session.mat \
  --save-path /absolute/path/to/results
```

### Option 2: Multiple files from a text file

Create a text file with one absolute `.mat` path per line, then run:

```bash
python main.py \
  --data-paths-file /absolute/path/to/data_paths.txt \
  --save-path /absolute/path/to/results
```

When running locally without Slurm, `SLURM_ARRAY_TASK_ID` defaults to `0`, so the code processes the first listed file.

## Running On PACE

### Step 1: Fill `data_paths.txt`

Edit [`data_paths.txt`](/home/ihsan/Desktop/data/Georgia_Tech/2AFC/Codes/Model_behavior/2AFC_computational_behavior_model_v20/data_paths.txt#L1) so it contains one absolute path per line:

```text
/storage/home/.../session_01.mat
/storage/home/.../session_02.mat
/storage/home/.../session_03.mat
```

Array task `0` will use line 1, task `1` will use line 2, and so on.

### Step 2: Update the Slurm array size

In [`Main.sh`](/home/ihsan/Desktop/data/Georgia_Tech/2AFC/Codes/Model_behavior/2AFC_computational_behavior_model_v20/Main.sh#L6), set:

```bash
#SBATCH --array=0-N
```

where `N = number_of_lines_in_data_paths.txt - 1`.

If you have 48 sessions, use:

```bash
#SBATCH --array=0-47
```

### Step 3: Check the PACE paths

Verify these lines in [`Main.sh`](/home/ihsan/Desktop/data/Georgia_Tech/2AFC/Codes/Model_behavior/2AFC_computational_behavior_model_v20/Main.sh#L17):

- the `cd` path points to your project folder on PACE
- the conda environment name is correct
- `DATA_PATHS_FILE` points to your real `data_paths.txt`
- `SAVE_PATH` points to your desired results directory

### Step 4: Submit the job

```bash
sbatch Main.sh
```

## How `main.py` Chooses Input Files

The input priority is:

1. `--data-path` arguments
2. `DATA_PATHS` environment variable
3. `--data-paths-file` or `DATA_PATHS_FILE`
4. legacy hardcoded fallback path in [`main.py`](/home/ihsan/Desktop/data/Georgia_Tech/2AFC/Codes/Model_behavior/2AFC_computational_behavior_model_v20/main.py#L65)

On PACE, the intended workflow is to use `DATA_PATHS_FILE`.

## Model Fitting Overview

Each session is fit independently with:

1. Stage 1: sensory-priority optimization
2. Stage 2: strategy-priority optimization
3. Stage 3: fine-tuning

The fitter uses multiple restarts and then chooses the best restart using:

- negative log-likelihood
- transition asymmetry loss

Key constants are defined in [`constants.py`](/home/ihsan/Desktop/data/Georgia_Tech/2AFC/Codes/Model_behavior/2AFC_computational_behavior_model_v20/constants.py#L1).

## Notes For HPC Stability

The code is already set up to be safer on shared CPU nodes:

- `torch` threads are limited to `SLURM_CPUS_PER_TASK`
- `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` are set
- multiprocessing uses the Slurm CPU allocation

This is important to avoid oversubscription and node contention on PACE.

## Quick Validation Before Submission

From the project folder, these are useful sanity checks:

```bash
python -m py_compile main.py constants.py Module/*.py
bash -n Main.sh
python main.py --help
```

## Common Failure Points

- `Task ID out of range`: your Slurm array is larger than the number of lines in `data_paths.txt`
- `No outcomes loaded. Check paths.`: one or more `.mat` paths are wrong, inaccessible, or not compatible
- `SessionData field not found`: the MATLAB file structure is different from the expected behavior export
- import errors on PACE: the conda environment is missing required Python packages
- wrong project path in `cd`: the Slurm script is pointing to the wrong folder

## Replication Note

This codebase was updated so the modular pipeline reproduces the execution path of the notebook rather than relying on notebook-only definitions. In particular:

- data-loading helpers were restored to the module layer
- notebook constants were restored to [`constants.py`](/home/ihsan/Desktop/data/Georgia_Tech/2AFC/Codes/Model_behavior/2AFC_computational_behavior_model_v20/constants.py#L1)
- the PACE entrypoint was made configurable through `data_paths.txt` and environment variables

## Recommended PACE Workflow

1. Copy the project folder to your PACE project directory.
2. Edit [`data_paths.txt`](/home/ihsan/Desktop/data/Georgia_Tech/2AFC/Codes/Model_behavior/2AFC_computational_behavior_model_v20/data_paths.txt#L1).
3. Update `#SBATCH --array` in [`Main.sh`](/home/ihsan/Desktop/data/Georgia_Tech/2AFC/Codes/Model_behavior/2AFC_computational_behavior_model_v20/Main.sh#L6).
4. Confirm the conda environment and `cd` path in [`Main.sh`](/home/ihsan/Desktop/data/Georgia_Tech/2AFC/Codes/Model_behavior/2AFC_computational_behavior_model_v20/Main.sh#L17).
5. Run the validation commands once.
6. Submit with `sbatch Main.sh`.
