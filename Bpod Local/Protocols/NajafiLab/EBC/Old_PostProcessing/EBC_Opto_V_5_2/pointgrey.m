% Define the paths to the FlyCapture2 SDK
sdkDir = 'C:\Program Files\Point Grey Research\FlyCapture2\';
includeDir = fullfile(sdkDir, 'include\C');
libDir = fullfile(sdkDir, 'lib64\C');

% Add the include directory to the MATLAB path
addpath(includeDir);

% Define the path to the FlyCapture2 library and header file
dllPath = fullfile(libDir, 'FlyCapture2_C_v100.dll');
headerPath = fullfile(includeDir, 'FlyCapture2_C.h');

% Load the FlyCapture2 library
if ~libisloaded('FlyCapture2_C')
    loadlibrary(dllPath, headerPath);
end

% Check if the library is loaded successfully
if libisloaded('FlyCapture2_C')
    disp('FlyCapture2 library loaded successfully.');
else
    error('Failed to load FlyCapture2 library.');
end