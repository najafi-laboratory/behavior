classdef SystemAudioPlayer < handle
    % Preload cues and play them through the current system audio output.
    properties (SetAccess = private)
        SamplingRate
    end

    properties (Access = private)
        Device
    end

    methods
        function obj = SystemAudioPlayer(samplingRate)
            PsychPortAudio('Verbosity', 0);
            InitializePsychSound(1);
            obj.SamplingRate = samplingRate;
            obj.Device = PsychPortAudio('Open', [], 1, 1, samplingRate, 2);
            PsychPortAudio('FillBuffer', obj.Device, zeros(2, 10));
        end

        function load(obj, waveform)
            if isvector(waveform)
                waveform = [waveform(:)'; waveform(:)'];
            end
            PsychPortAudio('FillBuffer', obj.Device, waveform);
        end

        function play(obj)
            PsychPortAudio('Start', obj.Device, 1, 0, 0);
        end

        function stop(obj)
            PsychPortAudio('Stop', obj.Device);
        end

        function delete(obj)
            if ~isempty(obj.Device)
                PsychPortAudio('Close', obj.Device);
                obj.Device = [];
            end
        end
    end
end
