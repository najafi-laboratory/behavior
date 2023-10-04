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


# current features

###
- read behavioral data from .mat as dict.
- fig1 shows the change of state percentage for completed trials across sessions. Green/light green are reward/naive reward, and red/light red are punish/naive punish. The x-axis is the seesion day and y-axis is the percentage. Four subplots correspond to four subjects.
- fig 2 is like fig 1 but gives the percentages of all outcomes.
- fig 3 shows the outcome percentages changing within a session. Trial outcomes are binned with some size and we compute the percentage for the trials within a bin. Each line gives one outcome.
- fig 4 is the psychometric plot. The xaxis is the stimulus interval and yaxis is the proability of right choice. Trials within a session have been binned and we calculate the mean for each bin. Different colors show different sessions.
- fig 5 is a sanity check for the stimulus interval. Each dot represents one isi, and all isi from one session has been merge. Dots are transparent, so when there are multiple dots in one position the color will be deeper.
- fig 6 checks the time difference between visual and auditory signal to make sure they are synchronized.


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
- Change isi reading structure from a list to a list of array.
- Delete trial duration.

## 2023.09.28
- DataIO now can read licking events during post-perturbation.
- Add a figure to plot psychometric of licking events during post-perturbation.

## 2023.09.29
- Change psychometric of licking events during post-perturbation for binned trials.
- Turn to plot VM1, VM4, VM5, and VM6.

## 2023.10.04
- Plot psychometric curves for each subjects.