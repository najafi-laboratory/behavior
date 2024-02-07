# file structure

- project folder
	- main.py : run this file to plot all figures
	- data
		- VMx
			- VMx_AFC_AV_LR_x_xxxxxxxx_xxxxxx : session data
	- figures
		- figx_xxxxx.pdf : figures
	- plot
		- figX.py code for each figure

# update note

## 2023.09.04
- First version.

## 2023.09.12
- Plot different subjects.

## 2023.09.15
- Merge states.
- Change reading structure from list to dictionaries.
- Now fig1 can plot number/percentage and set number as default.

## 2023.09.18
- Change fig1 layout and plot only the percentages for completed trials.
- Change data reading logic.
- Add a figure scatter plot for trial durations.
- Add a figure to check stim isi.

## 2023.09.19
- Use different colors for completed/incompleted trials in fig2.

## 2023.09.20
- Add a figure to check visual/auditory signal synchronization status.

## 2023.09.24
- Add a figure to check the percentages of all outcomes.
- Delete the trial duration plot and change figure ids.

## 2023.09.27
- Add a figure to check the percentage of completed trials with bins.
- Add subject name to read dictionaries.
- Change the figure title naming.
- Change ISI reading structure from a list to a list of array.
- Delete trial duration.

## 2023.09.28
- DataIO now can read licking events during post-perturbation.
- Add a figure to plot psychometric of licking events during post-perturbation.

## 2023.09.29
- Change psychometric of licking events during post-perturbation for binned trials.
- Turn to plot VM1, VM4, VM5, and VM6.

## 2023.10.04
- Plot psychometric curves for each subjects.

## 2023.10.07
- Add a figure for reaction time vs correctness.
- Add a figure for early choice vs ITI.

## 2023.10.08
- Change data reading structure.

## 2023.10.11
- The data folder should be specified in DataIO.

## 2023.10.15
- Corrected licking events start times.

## 2023.10.16
- Change fig4 and fig7 layouts.

## 2023.10.18
- Exchange fig1 and fig2.
- Separate licking events and first licking reactions.
- Add first reaction time vs probability of correctness.

## 2023.10.21
- Delete wrong reaction figure.
- Fix reaction time.

## 2023.10.29
- Add maximum showing session for AV checks.

## 2023.12.28
- Add reaction time across sessions.
- Add change of mind percentage.

## 2024.01.14
- Seperate figures for each subject.

## 2024.02.07
- Plot adjustablee figures for given subjects.