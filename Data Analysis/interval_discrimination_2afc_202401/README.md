# Usage

To plot all figires for a subject, in main.py at the subject name into subject_list, and then run main.py.


# file structure

- interval_discrimination_2afc_202401
	- main.py : run this file to plot all figures
	- DataIO.py : used in main.py to process bpod session data as mat files
	- session_data:
		- subject : folder for each subject
			- subject_iloveycforever_yymmdd_hhmmss.mat : bpod mat file data for each session
	- figures:
		- subject_report.pdf : each page has all figures for one subject
	- plot: folder for all plotting scripts
		- plot_outcome.py : all outcome percentage
		- plot_complete_trials.py : reward and punish percentage
		- plot_psychometric_post.py : psychometric function with post perturbation isi read from protocol
		- plot_psychometric_percep.py : psychometric function with post perturbation isi computed from BNC signals
		- plot_psychometric_pre.py : psychometric function with pre perturbation isi read from protocol
		- plot_psychometric_epoch.py : single session psychometric function with post perturbation isi read from protocol
		- plot_reaction_time.py : reaction time since stimulus onset
		- plot_reaction_outcome.py : reaction time for different choices and outcomes
		- plot_decision_time.py : reaction time since stimulus onset
		- plot_decision_outcome.py : reaction time for different choices and outcomes
	- backup : folder for legacy code in case

# update note

## 2024.05.30
- Separated from old analysis code.
- First release.

## 2024.06.19
- Added psychometric function for both empirical mean and perceptual mean.
- Now empirical mean is set to the mean of isi that was planned to play but not the continuous sampling value.
- Aborted trials when decision was made before the 1st perturbation stimuli.

