function SoftCodeHandler_PlayVideo(VidID)
global BpodSystem
if VidID ~= 255
    %BpodSystem.PluginObjects.V.tic1 = tic; %% need to factor in processing overhead from whence play is called
    BpodSystem.PluginObjects.V.play(VidID);
else
    BpodSystem.PluginObjects.V.stop;
end