%{
----------------------------------------------------------------------------

This file is part of the Sanworks Bpod repository
Copyright (C) 2022 Sanworks LLC, Rochester, New York, USA

----------------------------------------------------------------------------

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3.

This program is distributed  WITHOUT ANY WARRANTY and without even the 
implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
%}

function LastTrialEncoderPlot(axes, op, choiceThreshold, varargin)
global BpodSystem
switch op
    case 'init'
        BpodSystem.GUIHandles.EncoderPlot = plot(axes, 0,0, 'k-', 'LineWidth', 2);
        BpodSystem.GUIHandles.EncoderPlotThreshold1Line = line([0,1000],[0.0 0.0], 'Color', 'k', 'LineStyle', ':');
        BpodSystem.GUIHandles.EncoderPlotThreshold2Line = line([0,1000],[choiceThreshold choiceThreshold], 'Color', 'k', 'LineStyle', ':');        
       BpodSystem.GUIHandles.EncoderPlotThreshold3Line = line([0,1000],[0.2 0.2], 'Color', 'k', 'LineStyle', ':');        

        BpodSystem.GUIHandles.EncoderPlotSetLeverBeforePressStartLine = xline(0, '-', 'SetLeverBeforePress Start', 'Color', 'k', 'LineStyle', 'none');
        BpodSystem.GUIHandles.EncoderPlotVisStimTrigger1StartLine = xline(0, '-', 'VisStimTrigger1 Start', 'Color', 'k', 'LineStyle', 'none');
        BpodSystem.GUIHandles.EncoderPlotVisualStimulus11StartLine = xline(0, '-', 'VisualStimulus1 Start', 'Color', 'k', 'LineStyle', 'none');        
        BpodSystem.GUIHandles.EncoderPlotWaitForPress1StartLine = xline(0, '-', 'WaitForPress1 Start', 'Color', 'k', 'LineStyle', 'none');        
        BpodSystem.GUIHandles.EncoderPlotLeverRetract1StartLine = xline(0, '-', 'LeverRetract1 Start', 'Color', 'k', 'LineStyle', 'none');
        BpodSystem.GUIHandles.EncoderPlotRewardStartLine = xline(0, '-', 'Reward Start', 'Color', 'k', 'LineStyle', 'none');      
        BpodSystem.GUIHandles.EncoderPlotDidNotPress1StartLine = xline(0, '-', 'DidNotPress1 Start', 'Color', 'k', 'LineStyle', 'none');        
        BpodSystem.GUIHandles.EncoderPlotITIStartLine = xline(0, '-', 'ITI Start', 'Color', 'k', 'LineStyle', 'none');

        %set(BpodSystem.GUIHandles.EncoderPlotSetLeverBeforePressStartLine,'value',3, 'LineStyle', 'none');     

        set(axes, 'box', 'off', 'tickdir', 'out');
        ylabel('Position (deg)', 'FontSize', 12); 
        xlabel('Time (s)', 'FontSize', 12);

        % plot for zoomed view of lever return-> reward
        


    case 'update'
        EncoderData = varargin{1};
        TrialDuration = varargin{2};
   
        SetLeverBeforePressTimes = varargin{3};
        VisStimTrigger1Times = varargin{4};
        VisualStimulus1Times = varargin{5};
        WaitForPress1Times = varargin{6};

        LeverRetract1Times = varargin{7};
        Reward1Times = varargin{8};

        DidNotPress1Times = varargin{9};

        ITITimes = varargin{10};

        LeverResetPos = varargin{11};

        set(BpodSystem.GUIHandles.EncoderPlot, 'XData', EncoderData.Times,'YData', EncoderData.Positions);
        set(axes, 'ylim', [-1 choiceThreshold*4], 'xlim', [0 TrialDuration]);
        set(BpodSystem.GUIHandles.EncoderPlotThreshold1Line,'ydata',[0.0, 0.0]);
        set(BpodSystem.GUIHandles.EncoderPlotThreshold2Line,'ydata',[choiceThreshold, choiceThreshold]);  
        %set(BpodSystem.GUIHandles.EncoderPlotThreshold3Line,'ydata',[LeverResetPos(end), LeverResetPos(end)]);  

        %set(BpodSystem.GUIHandles.EncoderPlotSetLeverBeforePressStartLine,'value',SetLeverBeforePressTimes(1), 'LineStyle', ':');     
        %set(BpodSystem.GUIHandles.EncoderPlotVisStimTrigger1StartLine,'value',VisStimTrigger1Times(1), 'LineStyle', ':'); 
        set(BpodSystem.GUIHandles.EncoderPlotVisualStimulus11StartLine,'value',VisualStimulus1Times(1), 'LineStyle', ':'); 
        set(BpodSystem.GUIHandles.EncoderPlotWaitForPress1StartLine,'value',WaitForPress1Times(1), 'LineStyle', ':'); 
        
        if ~isnan(LeverRetract1Times(1)) && ~isnan(Reward1Times(1))
            set(BpodSystem.GUIHandles.EncoderPlotLeverRetract1StartLine,'value',LeverRetract1Times(1), 'LineStyle', ':'); 
            set(BpodSystem.GUIHandles.EncoderPlotRewardStartLine,'value',Reward1Times(1), 'LineStyle', ':'); 
        else
            set(BpodSystem.GUIHandles.EncoderPlotLeverRetract1StartLine,'LineStyle', 'none'); 
            set(BpodSystem.GUIHandles.EncoderPlotRewardStartLine,'LineStyle', 'none'); 
        end

        if ~isnan(DidNotPress1Times(1))
            set(BpodSystem.GUIHandles.EncoderPlotDidNotPress1StartLine,'value',DidNotPress1Times(1), 'LineStyle', ':');
        else
            set(BpodSystem.GUIHandles.EncoderPlotDidNotPress1StartLine,'LineStyle', 'none');
        end

        %set(BpodSystem.GUIHandles.EncoderPlotITIStartLine,'value',ITITimes(1), 'LineStyle', ':');


        % BpodSystem.GUIHandles.EncoderPlotSetLeverBeforePressStartLine = xline(SetLeverBeforePressTimes(1), '-', 'SetLeverBeforePress Start', 'Color', 'k', 'LineStyle', ':');
        % BpodSystem.GUIHandles.EncoderPlotVisStimTrigger1StartLine = xline(VisStimTrigger1Times(1), '-', 'VisStimTrigger1 Start', 'Color', 'k', 'LineStyle', ':');
        % BpodSystem.GUIHandles.EncoderPlotVisualStimulus11StartLine = xline(VisualStimulus1Times(1), '-', 'VisualStimulus1 Start', 'Color', 'k', 'LineStyle', ':');        
        % BpodSystem.GUIHandles.EncoderPlotWaitForPress1StartLine = xline(WaitForPress1Times(1), '-', 'WaitForPress1 Start', 'Color', 'k', 'LineStyle', ':');
        % 
        % if ~isnan(LeverRetract1Times(1)) && ~isnan(RewardTimes(1))
        %     BpodSystem.GUIHandles.EncoderPlotLeverRetract1StartLine = xline(LeverRetract1Times(1), '-', 'LeverRetract1 Start', 'Color', 'k', 'LineStyle', ':');
        %     BpodSystem.GUIHandles.EncoderPlotRewardStartLine = xline(RewardTimes(1), '-', 'Reward Start', 'Color', 'k', 'LineStyle', ':');
        % end
        % if ~isnan(DidNotPress1Times(1))
        %     BpodSystem.GUIHandles.EncoderPlotDidNotPress1StartLine = xline(DidNotPress1Times(1), '-', 'DidNotPress1 Start', 'Color', 'k', 'LineStyle', ':');
        % end
        % BpodSystem.GUIHandles.EncoderPlotITIStartLine = xline(ITITimes(1), '-', 'ITI Start', 'Color', 'k', 'LineStyle', ':');
        
        
        %BpodSystem.GUIHandles.EncoderPlotWaitForPress1EndLine = xline(WaitForPress1Times(2), '-', 'WaitForPress1 End', 'Color', 'k', 'LineStyle', ':');


%         SetLeverBeforePressTimes = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.SetLeverBeforePress;
%         VisStimTrigger1Times = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.VisStimTrigger1;
%         VisualStimulus1Times = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.VisualStimulus1;
%         WaitForPress1Times = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.WaitForPress1;
%         LeverRetract1Times = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.LeverRetract1;
%         RewardTimes = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.Reward;
%         ITITimes = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.ITI;
end