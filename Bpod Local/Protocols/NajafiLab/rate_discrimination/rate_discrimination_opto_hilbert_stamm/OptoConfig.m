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
                    end
            end
        end
    end
end
