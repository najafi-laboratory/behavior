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
        BpodSystem.GUIHandles.EncoderPlotThreshold3Line = line([0,1000],[0.3 0.3], 'Color', 'k', 'LineStyle', ':');        

        BpodSystem.GUIHandles.EncoderPlotPreVisStimITIStartLine = xline(0, '-', 'PreVisStimITI Start', 'Color', 'k', 'LineStyle', 'none');
        BpodSystem.GUIHandles.EncoderPlotVisStimTrigger1StartLine = xline(0, '-', 'VisDetect1 Start', 'Color', 'k', 'LineStyle', 'none');
        BpodSystem.GUIHandles.EncoderPlotVisualStimulus11StartLine = xline(0, '-', 'VisualStimulus1 Start', 'Color', 'k', 'LineStyle', 'none');        
        BpodSystem.GUIHandles.EncoderPlotWaitForPress1StartLine = xline(0, '-', 'WaitForPress1 Start', 'Color', 'r', 'LineStyle', 'none');        
        BpodSystem.GUIHandles.EncoderPlotLeverRetract1StartLine = xline(0, '-', 'LeverRetract1 Start', 'Color', 'k', 'LineStyle', 'none');
        BpodSystem.GUIHandles.EncoderPlotReward1StartLine = xline(0, '-', 'Reward1 Start', 'Color', 'k', 'LineStyle', 'none');      
        
        BpodSystem.GUIHandles.EncoderPlotDidNotPress1StartLine = xline(0, '-', 'DidNotPress1 Start', 'Color', 'k', 'LineStyle', 'none');        
        BpodSystem.GUIHandles.EncoderPlotITIStartLine = xline(0, '-', 'ITI Start', 'Color', 'k', 'LineStyle', 'none');

        BpodSystem.GUIHandles.EncoderPlotVisualStimulus12StartLine = xline(0, '-', 'VisualStimulus2 Start', 'Color', 'k', 'LineStyle', 'none');        
        BpodSystem.GUIHandles.EncoderPlotWaitForPress2StartLine = xline(0, '-', 'WaitForPress2 Start', 'Color', 'r', 'LineStyle', 'none');        
        BpodSystem.GUIHandles.EncoderPlotLeverRetract2StartLine = xline(0, '-', 'LeverRetract2 Start', 'Color', 'k', 'LineStyle', 'none');
        BpodSystem.GUIHandles.EncoderPlotReward2StartLine = xline(0, '-', 'Reward2 Start', 'Color', 'k', 'LineStyle', 'none');      
        BpodSystem.GUIHandles.EncoderPlotDidNotPress2StartLine = xline(0, '-', 'DidNotPress2 Start', 'Color', 'k', 'LineStyle', 'none');        

        BpodSystem.GUIHandles.EncoderPlotWaitForPress3StartLine = xline(0, '-', 'WaitForPress3 Start', 'Color', 'r', 'LineStyle', 'none');        
        BpodSystem.GUIHandles.EncoderPlotLeverRetract3StartLine = xline(0, '-', 'LeverRetract3 Start', 'Color', 'k', 'LineStyle', 'none');
        BpodSystem.GUIHandles.EncoderPlotReward3StartLine = xline(0, '-', 'Reward3 Start', 'Color', 'k', 'LineStyle', 'none');      
        BpodSystem.GUIHandles.EncoderPlotDidNotPress3StartLine = xline(0, '-', 'DidNotPress3 Start', 'Color', 'k', 'LineStyle', 'none');        

        BpodSystem.GUIHandles.EncoderPlotEarlyPressStartLine = xline(0, '-', 'Early Press Start', 'Color', 'r', 'LineStyle', 'none');        

        BpodSystem.GUIHandles.EncoderPlotRewardStartLine = xline(0, '-', 'Reward Start', 'Color', 'k', 'LineStyle', 'none');

        BpodSystem.GUIHandles.EncoderPlotPrePress2DelayStartLine = xline(0, '-', 'PrePress2Delay Start', 'Color', 'k', 'LineStyle', 'none');
        
        %set(BpodSystem.GUIHandles.EncoderPlotSetLeverBeforePressStartLine,'value',3, 'LineStyle', 'none');     

        set(axes, 'box', 'off', 'tickdir', 'out');
        ylabel('Position (deg)', 'FontSize', 12); 
        xlabel('Time (s)', 'FontSize', 12);

        % plot for zoomed view of lever return-> reward
        


    case 'update'
        EncoderData = varargin{1};
        TrialDuration = varargin{2};
   
        PreVisStimITITimes = varargin{3};
        VisStimTrigger1Times = varargin{4};
        VisualStimulus1Times = varargin{5};
        WaitForPress1Times = varargin{6};

        LeverRetract1Times = varargin{7};
        Reward1Times = varargin{8};

        DidNotPress1Times = varargin{9};

        ITITimes = varargin{10};

        LeverResetPos = varargin{11};

        WaitForPress2Times = varargin{12};
        LeverRetract2Times = varargin{13};
        Reward2Times = varargin{14};
        DidNotPress2Times = varargin{15};

        WaitForPress3Times = varargin{16};
        LeverRetract3Times = varargin{17};
        Reward3Times = varargin{18};
        DidNotPress3Times = varargin{19};

        RewardTimes = varargin{20};

        EarlyPressTimes = varargin{21};

        VisualStimulus2Times = varargin{22};

        PrePress2DelayTimes =  varargin{23};

        set(BpodSystem.GUIHandles.EncoderPlot, 'XData', EncoderData.Times,'YData', EncoderData.Positions);
        set(axes, 'ylim', [-1 choiceThreshold*4], 'xlim', [0 TrialDuration]);
        set(BpodSystem.GUIHandles.EncoderPlotThreshold1Line,'ydata',[0.0, 0.0]);
        set(BpodSystem.GUIHandles.EncoderPlotThreshold2Line,'ydata',[choiceThreshold, choiceThreshold]);  
        %set(BpodSystem.GUIHandles.EncoderPlotThreshold3Line,'ydata',[LeverResetPos(end), LeverResetPos(end)]);  

        %set(BpodSystem.GUIHandles.EncoderPlotSetLeverBeforePressStartLine,'value',PreVisStimITITimes(1), 'LineStyle', ':');     
        %set(BpodSystem.GUIHandles.EncoderPlotVisStimTrigger1StartLine,'value',VisStimTrigger1Times(1), 'LineStyle', ':'); 
        set(BpodSystem.GUIHandles.EncoderPlotVisualStimulus11StartLine,'value',VisualStimulus1Times(1), 'LineStyle', ':'); 
        set(BpodSystem.GUIHandles.EncoderPlotWaitForPress1StartLine,'value',WaitForPress1Times(1), 'LineStyle', ':'); 
        
        % if ~isnan(LeverRetract1Times(1)) && ~isnan(Reward1Times(1))
        %     set(BpodSystem.GUIHandles.EncoderPlotLeverRetract1StartLine,'value',LeverRetract1Times(1), 'LineStyle', ':'); 
        %     set(BpodSystem.GUIHandles.EncoderPlotRewardStartLine,'value',Reward1Times(1), 'LineStyle', ':'); 
        % else
        %     set(BpodSystem.GUIHandles.EncoderPlotLeverRetract1StartLine,'LineStyle', 'none'); 
        %     set(BpodSystem.GUIHandles.EncoderPlotRewardStartLine,'LineStyle', 'none'); 
        % end
        %% press 1

        if ~isnan(LeverRetract1Times(1))
            set(BpodSystem.GUIHandles.EncoderPlotLeverRetract1StartLine,'value',LeverRetract1Times(1), 'LineStyle', ':'); 
        else
            set(BpodSystem.GUIHandles.EncoderPlotLeverRetract1StartLine,'LineStyle', 'none'); 
        end

        if ~isnan(Reward1Times(1))
            set(BpodSystem.GUIHandles.EncoderPlotReward1StartLine,'value',Reward1Times(1), 'LineStyle', ':'); 
        else
            set(BpodSystem.GUIHandles.EncoderPlotReward1StartLine,'LineStyle', 'none'); 
        end

        if ~isnan(DidNotPress1Times(1))
            set(BpodSystem.GUIHandles.EncoderPlotDidNotPress1StartLine,'value',DidNotPress1Times(1), 'LineStyle', ':');
        else
            set(BpodSystem.GUIHandles.EncoderPlotDidNotPress1StartLine,'LineStyle', 'none');
        end

        %% press 2

        if ~isnan(PrePress2DelayTimes(1))
            set(BpodSystem.GUIHandles.EncoderPlotPrePress2DelayStartLine,'value',PrePress2DelayTimes(1), 'LineStyle', ':'); 
        else
            set(BpodSystem.GUIHandles.EncoderPlotPrePress2DelayStartLine,'LineStyle', 'none');
        end

        if ~isnan(VisualStimulus2Times(1))
            set(BpodSystem.GUIHandles.EncoderPlotVisualStimulus12StartLine,'value',VisualStimulus2Times(1), 'LineStyle', ':'); 
        else
            set(BpodSystem.GUIHandles.EncoderPlotVisualStimulus12StartLine,'LineStyle', 'none');
        end

        if ~isnan(WaitForPress2Times(1))
            set(BpodSystem.GUIHandles.EncoderPlotWaitForPress2StartLine,'value',WaitForPress2Times(1), 'LineStyle', ':'); 
        else
            set(BpodSystem.GUIHandles.EncoderPlotWaitForPress2StartLine,'LineStyle', 'none');
        end

        if ~isnan(LeverRetract2Times(1))
            set(BpodSystem.GUIHandles.EncoderPlotLeverRetract2StartLine,'value',LeverRetract2Times(1), 'LineStyle', ':'); 
        else
            set(BpodSystem.GUIHandles.EncoderPlotLeverRetract2StartLine,'LineStyle', 'none'); 
        end

        if ~isnan(Reward2Times(1))
            set(BpodSystem.GUIHandles.EncoderPlotReward2StartLine,'value',Reward2Times(1), 'LineStyle', ':'); 
        else
            set(BpodSystem.GUIHandles.EncoderPlotReward2StartLine,'LineStyle', 'none'); 
        end

        if ~isnan(DidNotPress2Times(1))
            set(BpodSystem.GUIHandles.EncoderPlotDidNotPress2StartLine,'value',DidNotPress2Times(1), 'LineStyle', ':');
        else
            set(BpodSystem.GUIHandles.EncoderPlotDidNotPress2StartLine,'LineStyle', 'none');
        end

        %% press 3  

        if ~isnan(WaitForPress3Times(1))
            set(BpodSystem.GUIHandles.EncoderPlotWaitForPress3StartLine,'value',WaitForPress3Times(1), 'LineStyle', ':'); 
        else
            set(BpodSystem.GUIHandles.EncoderPlotWaitForPress3StartLine,'LineStyle', 'none');
        end

        if ~isnan(LeverRetract3Times(1))
            set(BpodSystem.GUIHandles.EncoderPlotLeverRetract3StartLine,'value',LeverRetract3Times(1), 'LineStyle', ':'); 
        else
            set(BpodSystem.GUIHandles.EncoderPlotLeverRetract3StartLine,'LineStyle', 'none'); 
        end

        if ~isnan(Reward3Times(1))
            set(BpodSystem.GUIHandles.EncoderPlotReward3StartLine,'value',Reward3Times(1), 'LineStyle', ':'); 
        else
            set(BpodSystem.GUIHandles.EncoderPlotReward3StartLine,'LineStyle', 'none'); 
        end

        if ~isnan(DidNotPress3Times(1))
            set(BpodSystem.GUIHandles.EncoderPlotDidNotPress3StartLine,'value',DidNotPress3Times(1), 'LineStyle', ':');
        else
            set(BpodSystem.GUIHandles.EncoderPlotDidNotPress3StartLine,'LineStyle', 'none');
        end

        %% Early Press

        if ~isnan(EarlyPressTimes(1))
            set(BpodSystem.GUIHandles.EncoderPlotEarlyPressStartLine,'value',EarlyPressTimes(1), 'LineStyle', ':'); 
        else
            set(BpodSystem.GUIHandles.EncoderPlotEarlyPressStartLine,'LineStyle', 'none'); 
        end


        %% Reward

        if ~isnan(RewardTimes(1))
            set(BpodSystem.GUIHandles.EncoderPlotRewardStartLine,'value',RewardTimes(1), 'LineStyle', ':'); 
        else
            set(BpodSystem.GUIHandles.EncoderPlotRewardStartLine,'LineStyle', 'none'); 
        end

        %% ITI

        set(BpodSystem.GUIHandles.EncoderPlotITIStartLine,'value',ITITimes(1), 'LineStyle', ':');


        % BpodSystem.GUIHandles.EncoderPlotSetLeverBeforePressStartLine = xline(PreVisStimITITimes(1), '-', 'SetLeverBeforePress Start', 'Color', 'k', 'LineStyle', ':');
        % BpodSystem.GUIHandles.EncoderPlotVisStimTrigger1StartLine = xline(VisStimTrigger1Times(1), '-', 'VisDetect1 Start', 'Color', 'k', 'LineStyle', ':');
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


%         PreVisStimITITimes = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.SetLeverBeforePress;
%         VisStimTrigger1Times = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.VisDetect1;
%         VisualStimulus1Times = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.VisualStimulus1;
%         WaitForPress1Times = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.WaitForPress1;
%         LeverRetract1Times = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.LeverRetract1;
%         RewardTimes = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.Reward;
%         ITITimes = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.ITI;
end