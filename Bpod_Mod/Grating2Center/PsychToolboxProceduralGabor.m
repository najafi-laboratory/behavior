% This class can wrap PsychToolbox support for rendering a full-screen procedurally defined gabor patch

% Usage:
% G = PsychToolboxProceduralGabor(ScreenID);
% Render:
% G.draw(Contrast, Angle, PositionOffset);
% Black screen:
% G.clear;

classdef PsychToolboxProceduralGabor < handle
    properties

    end
    properties (Access = private)
        Window % PsychToolbox Window object
        BlankScreen % A black screen matching the size of the monitor
        WindowSize
        GaborTexture
        GaborRectangle
        GaborCenterOffset
    end
    methods  
        function obj = PsychToolboxProceduralGabor(ScreenID)
            Screen('Preference','SkipSyncTests', 1);
            [obj.WindowSize(1), obj.WindowSize(2)]=Screen('WindowSize', ScreenID);
            obj.GaborCenterOffset = (obj.WindowSize(2)-((obj.WindowSize(1))/2))/2;
            obj.Window = Screen('OpenWindow', ScreenID, 127);
            Frame = ones(obj.WindowSize(2), obj.WindowSize(1))*127;
            obj.BlankScreen = Screen('MakeTexture', obj.Window, Frame);
            Screen('DrawTexture', obj.Window, obj.BlankScreen);
            Screen('Flip', obj.Window);
            [obj.GaborTexture, obj.GaborRectangle] = CreateProceduralGabor(obj.Window, obj.WindowSize(2), obj.WindowSize(2), [], [0.5,0.5,0.5,0]);
        end
        function draw(obj,Contrast, Angle, PositionOffset)
            dstRect = OffsetRect(obj.GaborRectangle, PositionOffset+obj.GaborCenterOffset, 0);
            Screen('DrawTexture', obj.Window, obj.GaborTexture, [], dstRect,...
            Angle,... % Tilt
            [], [], [], [], kPsychDontDoRotation,...
            [0, ... % Phase
            0.01,... % Frequency
            200, ... % SC
            (Contrast/100)*128 ...
            1, 0, 0, 0]);
            Screen('Flip', obj.Window);
        end
        function clear(obj)
            Screen('DrawTexture', obj.Window, obj.BlankScreen);
            Screen('Flip', obj.Window);
        end
        function delete(obj)
            Screen('CloseAll');
        end
    end
end