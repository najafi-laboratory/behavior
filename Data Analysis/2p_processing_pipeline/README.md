# file structure


# update note

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