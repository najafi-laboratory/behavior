% C:\behavior\behavior_setup\Bpod Local\Protocols\NajafiLab\Joystick\Development\DoricSystemDLL\API\lib\x64\release

dllPath = 'C:\DoricLaserAPI\DoricSystemDLL\API\lib\x64\release\DoricSystem.dll';
% headerPath = 'C:\DoricLaserAPI\DoricSystemDLL\API\lib\x64\release\DoricSystem.h';

headerPath = 'C:\DoricLaserAPI\DoricSystemDLL\API\include\doric_system_wrapper.h';

% Load the library (if you have the header file)
% loadlibrary('DoricSystem.dll', 'DoricSystem.h')
% loadlibrary(dllPath, headerPath);

% loadlibrary('C:\DoricLaserAPI\DoricSystemDLL\API\lib\x64\release\DoricSystem.dll')

% dllPath = 'C:\DoricLaserAPI\DoricSystemDLL\API\lib\x64\release\DoricSystem.dll';
% loadlibrary(dllPath, '', 'alias', 'DoricSystem')


% libfunctions('DoricSystem', '-full')


% NET.addAssembly('DoricLaser.dll');
% NET.addAssembly(dllPath);
% obj = DoricLaser.ClassName();
% obj.MethodName(args);

pythonDoricPath = 'C:\DoricLaserAPI\DoricSystemDLL\Examples\Python\LightSource';
insert(py.sys.path, int32(0), pythonDoricPath);  % Add Python script location
py.importlib.import_module('light_source_main'); % Import the Python script
result = py.light_source_main.main();   % Call the Python function
disp(result);

if 0 

    tic
    result = py.light_source_main.SetLaserSequence();   % Call the Python function
    disp(result);
    toc

    result = py.light_source_main.DisconnectLaser();   % Call the Python function
    disp(result);
end

