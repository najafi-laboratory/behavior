function SoftCodeHandler_Protocol(code)
global BpodSystem

switch code
    case 1
        playVideo(1);
    case 2
        playVideo(2);
    case 3
        showGray;
end

    function playVideo(index)
        stopVideo;
        BpodSystem.PluginObjects.V.play(index);
    end

    function showGray
        stopVideo;
        BpodSystem.PluginObjects.V.setSyncPatch(0);
    end

    function stopVideo
        try
            BpodSystem.PluginObjects.V.stop;
        catch exception
            if ~contains(exception.message, 'not running')
                rethrow(exception)
            end
        end
    end
end
