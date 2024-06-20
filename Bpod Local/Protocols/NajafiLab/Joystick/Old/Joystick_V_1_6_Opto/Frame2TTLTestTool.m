classdef Frame2TTLTestTool < handle
    
    properties
        V % Video server
        SyncPatchIntensity = 255;
    end

    properties (Access = private)
        TestVideo
    end

    methods
        function obj = Frame2TTLTestTool
            obj.V = PsychToolboxVideoPlayer(2, [0 0], [0 0], [60 60], 0);
            maxSecondsPerTrigger = 1;
            obj.TestVideo = (ones(obj.V.ViewPortDimensions(2),obj.V.ViewPortDimensions(1), 60*maxSecondsPerTrigger)*obj.SyncPatchIntensity);
            obj.V.loadVideo(1, obj.TestVideo);
        end
        function set.SyncPatchIntensity(obj, newIntensity)
            obj.V.SyncPatchIntensity = newIntensity;
            obj.V.loadVideo(1, obj.TestVideo);
            obj.SyncPatchIntensity = newIntensity;
        end
        function setPatch(obj, Level)
            obj.V.setSyncPatch(Level);
        end
        function playVideo(obj)
            obj.V.play(1);
        end
    end
end