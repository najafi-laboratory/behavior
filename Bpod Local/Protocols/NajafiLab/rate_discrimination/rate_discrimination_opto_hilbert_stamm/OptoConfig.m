classdef OptoConfig
   properties
      EnableOpto = 0;
   end
    methods        
        function obj = OptoConfig(EnableOpto)
            if nargin == 1
                obj.EnableOpto = EnableOpto;
            end
        end
        function [AudStimOpto] = GetAudStimOpto(obj, OptoTrialType)
            switch obj.EnableOpto
                case 0
                    AudStimOpto = {'HiFi1', ['P', 4]};
                case 1
                    if OptoTrialType == 2
                        AudStimOpto = {'HiFi1', ['P', 4], 'GlobalTimerTrig', 1};
                    else
                        AudStimOpto = {'HiFi1', ['P', 4]};
                    end
            end
        end
        function [sma] = InsertGlobalTimer(obj, sma, VisStim)
            if obj.EnableOpto
                sma = SetGlobalTimer(sma, 'TimerID', 1, 'Duration', VisStim.VisStimDuration, 'OnsetDelay', 0,...
                     'Channel', 'BNC2', 'OnLevel', 1, 'OffLevel', 0,...
                     'Loop', 1, 'SendGlobalTimerEvents', 0, 'LoopInterval', VisStim.Grating.Dur,...
                     'GlobalTimerEvents', 0, 'OffsetValue', 0);
            end
        end
    end
end
