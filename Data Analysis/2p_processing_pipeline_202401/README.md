# Usage

Run on windows command line window:
```
python main.py `
--denoise 0 `
--spatial_scale 2 `
--run_Plotter 1 `
--data_path './testdata/C2' `
--save_path0 './results/C2' `
--nchannels 1 `
--functional_chan 2 `
--brain_region 'crbl' `
```
Run on linux server:
```
sbatch run.sh
```

# Update note

## 2024.01.19
- First release.

## 2024.01.25
- Added comments.
- Now neural_trial.h5 saves raw recordings. 
- Alignment works correctly confirmed.
- Move config.json out from the folder.

## 2024.01.27
- Adjusted plot codes structure.
- Plot masks figure completed.
- Plot example traces completed.
- Plot stimulus average response completed.
- Plot grand average across stimulus and neurons completed.
- Rewrote code for getting mean images after registration.
- Added argparse support for specifying paths.

## 2024.01.28
- Plot omission alignment and trial average completed.

## 20224.01.29
- Plot mask with different colors specifying channels completed.
- Plot interval distribution completed.
- Plot mean raw traces completed.
- Added args support for specifying channels.

## 2024.01.31
- Find rising and dropping edges changed to pre-append one 0 and same indice.
- Plot reference masks with channel colors.
- Adjusted line weight and scale for raw traces plot.
- Added a tentative function to hard code trial types via variance detection.
- Save voltage timestamps into the output file.
- Plot grating alignment completed.
- Plot omission alignment average completed.
- Plot pre-post alignment average completed.

## 2024.02.01
- Added error handling for data without voltage input.
- Added diameter control support for cellpose.
- Added number of channels in configuration.
- Read channel files with dynamic concatenate to lower memory usage.
- Read channel files with memory mapping to lower memory usage.

## 2024.02.02
- Rearranged file structure.
- Rewrote data normalization for figures.

## 2024.02.05
- Cell detection now uses costumized flow_threshold in ops.npy.
- Cellpose pre trained model now uses cyto2.
- Now signal extraction uses the masks of the max projection of functional channel.
- Completed computing masks overlap between channels to label neurons.
- Modified plotting masks with labels.
- Plot traces with different colors to specify excitory and inhibitory neurons.

## 2024.02.08
- Added voltage recording issue handler.

## 2024.02.10
- Deleted cellpose results on reference image.
- Added comparison between max projection and reference image for cellpose on functional channel.
- Plot spike trigger average completed.

## 2024.02.14
- Added args for cerebellum/ppc.
- Added functional ROI detection for cerebellum.
- Fixed bugs in plotting figures.
- Added standard error for average figures.
- Adjusted figure scale.

## 2024.02.16
- Change spike trigger average figures from individual traces to standard error.

## 2024.02.26
- Changed to use the signal density ratio between ROI and surroundings to classify inhibitory neurons.

## 2024.02.28
- Df/f computation and normalization with individual variance completed.
- Spike triggered average now takes global variance and individual means to find the threshold.
- Added reading bpod session data mat files.

## 2024.03.01
- Divided configuration json fiiles for cerebellum and ppc.
- Added backup folder to save unsued but potentially useful codes.
- Deleted cellpose for functional ROIs detection and diameter parameters.
- Turn to suite2p detection instead.
- Added exception handler for removing registration binary files.
- Deleted results retrieval in the main process.
- Deleted normalization on inhibitory neuron identification.

## 2024.03.03
- Moved spike threshold out of extraction module.
- Fine tuned parameters for ppc configurations.

## 2024.03.04
- Rewrite results retrieval module to post process module.
- Moved spike deconvolution and traces normalization to post process module.
- Merged signal synchronization to post processing.
- Deleted argument inputs for controlling individual modules.
- Removed a slash in sbatch script that may case bus error.

## 2024.03.19
- Now run rigid motion correction by default.
- Now tiff files after motion correction will be saved in temp folder.
- Added spatial signal correlation in inhibitory neuron identification.
- Rewrote configuration files and parameters.
- Now before motion correction the concatenate tiff file will be saved.
- Rewrote motion correction function to correct the workflow.
- Now run signal extrtaction only on functional channel.
- Now spatial scale for ROI detection can be set as an argument.
- Functions for post processing are separated into different files.