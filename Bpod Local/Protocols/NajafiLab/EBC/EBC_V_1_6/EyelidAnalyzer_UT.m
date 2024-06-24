MEV = EyelidAnalyzer;


MEV.startGUIVideo();


MEV.stopGUIVideo;


MEV.connectVideoTrial;
pause(0.1);
MEV.startVideoTrial;


%MEV.vid
MEV.vid.FramesAcquired
MEV.vid.FramesAvailable

isrunning(MEV.vid)


% Display all valid trigger configurations.
%triggerinfo(MEV.vid)
%MEV.src.TriggerMode