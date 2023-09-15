global BpodSystem

S = struct;

if isempty(fieldnames(S))  % If settings file was an empty struct, populate struct with default settings
    %% ITI params
    S.GUI.ITImin_s = 1;    % Minimum ITI (in seconds)
    S.GUI.ITImax_s = 5;    % Maximum ITI (in seconds)
    S.GUI.ITIlambda = 0.3;  % ITIlambda parameter of the exponential distribution
    S.GUIPanels.ITI_Dist = {'ITImin_s', 'ITImax_s', 'ITIlambda'};

    %% sound params
    S.GUI.DigitalAttenuation_dB = -20; % volume control: H.DigitalAttenuation_dB = -40;
    S.GUI.MasterVolume_percent = 0.5;    % volume control
    S.GUIPanels.Sound = {'DigitalAttenuation_dB', 'MasterVolume_percent'};

    %% init cue params
    S.GUI.InitCueVolume_percent = 0.5;  % volume control
    S.GUI.InitCueDuration_s = 0.05; % Duration of init sound
    S.GUI.InitWindowTimeout_s = 5; % How long the mouse has to initiate stimulus or miss init lick
    %S.GUI.InitCueFreq_Hz = 500; % Frequency of init cue
    %S.GUI.InitCueFreq_Hz = 525; % Frequency of init cue, even multiple of 44100 SF
    S.GUI.InitCueFreq_Hz = 4900; % Frequency of init cue, even multiple of 44100 SF
    S.GUIPanels.InitCue = {'InitCueVolume_percent', 'InitCueDuration_s', 'InitWindowTimeout_s', 'InitCueFreq_Hz'};

    %% go cue params
    S.GUI.GoCueVolume_percent = 0.5;  % volume control
    S.GUI.GoCueDuration_s = 0.05; % Duration of go sound
    %S.GUI.GoCueFreq_Hz = 2000; % Frequency of go cue
    %S.GUI.GoCueFreq_Hz = 2100; % Frequency of go cue, even multiple of 44100 SF
    S.GUI.GoCueFreq_Hz = 11025; % Frequency of go cue, even multiple of 44100 SF
    S.GUIPanels.GoCue = {'GoCueVolume_percent', 'GoCueDuration_s', 'GoCueFreq_Hz'};

    %% training level params
    S.GUI.TrainingLevel = 1; % Configurable training and test schemes.
                             % 1 - 'Naive', 2 - 'Trained'
    S.GUIMeta.TrainingLevel.Style = 'popupmenu'; % the GUIMeta field is used by the ParameterGUI plugin to customize UI objects.
    S.GUIMeta.TrainingLevel.String = {'Naive', 'Mid Trained', 'Well Trained'};
    S.GUI.NumEasyWarmupTrials = 20;
    S.GUIPanels.Training = {'TrainingLevel', 'NumEasyWarmupTrials'};

    %% difficulty params
    % percentage of full perturbation range as boundaries for difficulty levels
    S.GUI.PercentTrialsEasy = 100;
    S.GUI.PercentTrialsMediumEasy = 0;
    S.GUI.PercentTrialsMediumHard = 0;
    S.GUI.PercentTrialsHard = 0;    
    S.GUIPanels.Difficulty = {'PercentTrialsEasy', 'PercentTrialsMediumEasy', 'PercentTrialsMediumHard', 'PercentTrialsHard'};

    %% audio stim
    S.GUI.AudioStimEnable = 1;
    S.GUIMeta.AudioStimEnable.Style = 'checkbox';
    S.GUI.AudioStimVolume_percent = 0.5;  % volume control
    %S.GUI.AudioStimFreq_Hz = 15000; % Frequency of audio stim
    S.GUI.AudioStimFreq_Hz = 14700; % Frequency of audio stim, even multiple of SF = 44100
    S.GUIPanels.AudioStim = {'AudioStimEnable', 'AudioStimVolume_percent', 'AudioStimFreq_Hz'};

    %% vis stim params
    S.GUI.VisStimEnable = 1;
    S.GUIMeta.VisStimEnable.Style = 'checkbox';
    %S.GUI.GratingDur_s = 0.25; % Duration of grating stimulus in seconds - ORIGINAL
    S.GUI.GratingDur_s = 0.1; % Duration of grating stimulus in seconds - UPDATE
    %S.GUI.ISIOrig_s = 0.75; % Duration of *fixed* gray screen stimulus in seconds - ORIGINAL
    S.GUI.ISIOrig_s = 0.5; % Duration of *fixed* gray screen stimulus in seconds - UPDATE
    S.GUI.ExtraStimDurPostRew_Naive_s = 5; % naive mouse sees stimulus for this time (sec) after correct lick    
    S.GUI.NumISIOrigRep = 5; % number of grating/gray repetitions for vis stim first segment prior to perturbation
    S.GUI.PostPerturbDurMultiplier = 1.5; % scaling factor for post perturbation stimulus (postperturb = preperturb * PostPerturbDurMultiplier)    
    S.GUI.MinISIPerturb_ms = 100; % min time in ms for perturbation range from grating
    S.GUI.PreVisStimDelay_s = 0; % How long the mouse must poke in the center to activate the goal port
    S.GUI.PreGoCueDelay_s = 0;
    S.GUIPanels.VisStim = {'VisStimEnable', 'GratingDur_s', 'ISIOrig_s', 'ExtraStimDurPostRew_Naive_s', 'NumISIOrigRep', 'PostPerturbDurMultiplier', 'MinISIPerturb_ms', 'PreVisStimDelay_s', 'PreGoCueDelay_s'}; 
 
    %% contingency and bias params
    S.GUI.Short_Fast_ISIChoice = 1;   % set short ISI association to left or right side
                                % 1 - 'Left', 2 - 'Right'
    S.GUIMeta.Short_Fast_ISIChoice.Style = 'popupmenu'; % the GUIMeta field is used by the ParameterGUI plugin to customize UI objects.
    S.GUIMeta.Short_Fast_ISIChoice.String = {'Left', 'Right'};
    S.GUI.ShortISIFraction = 0.5;   % set fraction of trials that are short ISI (long ISI fraction = (1 - short))                                
    S.GUI.ManualSideSelect = 0;   % override to disable/enable manual selection of left/right for next trial
    S.GUIMeta.ManualSideSelect.Style = 'checkbox';
    S.GUI.ManualSide = 1;   % manual selection of left/right for next trial
    S.GUIMeta.ManualSide.Style = 'popupmenu'; % the GUIMeta field is used by the ParameterGUI plugin to customize UI objects.
    S.GUIMeta.ManualSide.String = {'Left', 'Right'};
    S.GUIPanels.Contingency_Bias = {'Short_Fast_ISIChoice', 'ShortISIFraction', 'ManualSideSelect', 'ManualSide'};

    %% reward params
    S.GUI.LeftValveTime_s = 0.25;
    S.GUI.RightValveTime_s = 0.25;
    S.GUI.CenterValveTime_s = 0.10;
    S.GUI.WindowRewardGrabDuration_Naive_s = 10;  % naive mouse has up to x seconds to grab reward    
    S.GUI.RewardDelay_s = 0; % How long the mouse must wait in the goal port for reward to be delivered
    S.GUI.EnableCenterLick_Trained = 0; % override to disable/enable center lick for well trained
    S.GUIMeta.EnableCenterLick_Trained.Style = 'checkbox';
    S.GUI.CenterLickWindow_Trained_s = 2;   
    S.GUIPanels.Reward = {'LeftValveTime_s', 'RightValveTime_s', 'CenterValveTime_s', 'WindowRewardGrabDuration_Naive_s', 'RewardDelay_s', 'EnableCenterLick_Trained', 'CenterLickWindow_Trained_s'};

    %% punish params
    S.GUI.IncorrectSoundVolume_percent = 0.15;  % volume control
    S.GUI.PunishSoundDuration_s = 1; % Seconds to wait on errors before next trial can start
    S.GUI.IncorrectSound = 1; % if 1, plays a white noise pulse on error. if 0, no sound is played.
    S.GUIMeta.IncorrectSound.Style = 'checkbox';
    S.GUIPanels.Punish = {'IncorrectSoundVolume_percent', 'PunishSoundDuration_s', 'IncorrectSound'}; 

    %% choice params    
    S.GUI.ChoiceWindow_s = 5; % How long after go cue until the mouse must make a choice
    S.GUI.ConfirmLickInterval_s = 0.2; % min interval until choice can be confirmed    
    S.GUI.ChoiceConfirmWindow_s = 5; % time during which correct choice can be confirmed    
    S.GUIPanels.Choice = {'ChoiceWindow_s', 'ConfirmLickInterval_s', 'ChoiceConfirmWindow_s'};

    %% temp debugging params
    % S.GUI.WindowRewardGrabDuration_Naive_s = 5;  % naive mouse has up to x seconds to grab reward
    % S.GUI.ChoiceWindow_s = 2; % How long after go cue until the mouse must make a choice
    % S.GUI.NumEasyWarmupTrials = 0;
    
    % S.GUI.ITImin_s = 0;    % Minimum ITI (in seconds)
    % S.GUI.ITImax_s = 0;    % Maximum ITI (in seconds)
end



BpodSystem.PluginObjects.V = [];
MonitorID = 2;
BpodSystem.PluginObjects.V = PsychToolboxVideoPlayer(MonitorID, 0, [0 0], [180 180], 0, 0); % Assumes second monitor is screen #2. Sync patch = 180x180 pixels
BpodSystem.PluginObjects.V.SyncPatchIntensity = 255; % increased, seems 140 doesn't always trigger BNC high

Ysize = BpodSystem.PluginObjects.V.ViewPortDimensions(2);
Xsize = BpodSystem.PluginObjects.V.ViewPortDimensions(1);

% compute grating according to square grid of largest pixel dimension
if Ysize > Xsize
    gratingSize = [Ysize, Ysize]; % Size of grating in pixels
else
    gratingSize = [Xsize, Xsize]; % Size of grating in pixels
end

spatialFreq = .005; % Spatial frequency of grating in cycles per pixel
% spatialFreq = .32; % Spatial frequency of grating in cycles per pixel
orientation = 0; % Orientation of grating in degrees
contrast = 1; % Contrast of grating (0 to 1)
phase = 0.5;

% Calculate the parameters needed for the grating
pixPerCycle = 1 / spatialFreq;
freqPerPixel = 1 / pixPerCycle;

[x, y] = meshgrid(1:gratingSize(1), 1:gratingSize(2));

gray = 0.5 * ones(gratingSize);
sinGrating = gray + contrast/2 .* sin(freqPerPixel * 2 * pi * (cos(orientation * pi / 180) * x + sin(orientation * pi / 180) * y) + phase);
sinGrating(sinGrating > 1) = 1; % Cap values above 1 to 1 (white)
sinGrating(sinGrating < 0) = 0; % Cap values below 0 to 0 (black)

gray = gray(1:Ysize, 1:Xsize); % clip to monitor
sinGrating = sinGrating(1:Ysize, 1:Xsize); % clip to monitor

% these images are 0 - 1 amplitude, fine for grating_flashes.m file but need to be 0 - 255 for
% building video this way
gray = gray * 255;
sinGrating = sinGrating * 255;

% S.GUI.GratingDur_s = 2.00; % Duration of grating stimulus in seconds
% S.GUI.ISIOrig_s = 1.0; % Duration of *fixed* gray screen stimulus in seconds

FramesPerSecond = BpodSystem.PluginObjects.V.DetectedFrameRate;

GratingDuration = S.GUI.GratingDur_s; % set duration of grating to stimulus interval
GrayFixedDuration = S.GUI.ISIOrig_s; % set duration of gray screen to inter stimulus interval

% GratingFrames = FramesPerSecond * GratingDuration;
% GrayFixedFrames = FramesPerSecond * GrayFixedDuration;

% GratingDuration = 0.09;
% GrayFixedDuration = 0.11;

% need an integer number of frames, there is no fractional frame
% need an even number of frames for sync patch to alternate
GratingFrames = convergent(FramesPerSecond * GratingDuration);  % maybe use floor for this? then continue to round up below?
if (mod(GratingFrames, 2) ~= 0)
    GratingFrames = GratingFrames + 1; % round up to nearest even integer
end
GratingDuration = GratingFrames / FramesPerSecond; % convert even rounded number of frames back into duration to calculate video duration

GrayFixedFrames = convergent(FramesPerSecond * GrayFixedDuration);
if (mod(GrayFixedFrames, 2) ~= 0)
    GrayFixedFrames = GrayFixedFrames + 1; % round up to nearest even integer
end
GrayFixedDuration = GrayFixedFrames / FramesPerSecond; % convert even rounded number of frames back into duration to calculate video duration

    % GrayPerturbFrames = convergent(FramesPerSecond * GrayPerturbDuration);
    % if (mod(GrayPerturbFrames, 2) ~= 0)
    %     GrayPerturbFrames = GrayPerturbFrames + 1; % round up to nearest even integer
    % end

 
VideoGrating = repmat(sinGrating, 1, 1, 2); % 2 frames to get sync signal encoded
VideoGrayFixed = repmat(gray, 1, 1, 2); % 2 frames to get sync signal encoded

BpodSystem.PluginObjects.V.loadVideo(1, VideoGrating);
BpodSystem.PluginObjects.V.loadVideo(2, VideoGrayFixed);

% compose grating video
GratingFrame_SyncW = BpodSystem.PluginObjects.V.Videos{1}.Data(1);
GratingFrame_SyncBlk = BpodSystem.PluginObjects.V.Videos{1}.Data(2);
GratingBlank = BpodSystem.PluginObjects.V.Videos{1}.Data(3);

%GratingPattern = [GratingFrame_SyncW GratingFrame_SyncBlk];
GratingPattern = [GratingFrame_SyncW GratingFrame_SyncW];

%GratingVideo = [repmat(GratingPattern, 1, GratingFrames/2) GratingBlank];
GratingVideo = [repmat(GratingPattern, 1, GratingFrames/2)];

BpodSystem.PluginObjects.V.Videos{3} = struct;
BpodSystem.PluginObjects.V.Videos{3}.nFrames = GratingFrames + 1; % + 1 for final frame
BpodSystem.PluginObjects.V.Videos{3}.Data = [GratingVideo GratingBlank];

% Indicate loading
BpodSystem.PluginObjects.V.loadText(1, 'Loading...', '', 80);
BpodSystem.PluginObjects.V.play(1);

BpodSystem.PluginObjects.V.TimerMode = 0;
pause(1.0); % matlab seems to require a pause here before clearing screen with play(0), 
            % otherwise can get stuck on Psychtoolbox splash screen
            % might need longer delay if purple image hangs on window open
BpodSystem.PluginObjects.V.play(0);
BpodSystem.SoftCodeHandlerFunction = 'SoftCodeHandler_PlayVideo';

%BpodSystem.PluginObjects.V.play(6);
%BpodSystem.PluginObjects.V.play(3);

%BpodSystem.PluginObjects.V.TimerMode = 2;
waitframes = 1;


Port = 5;
T = TCPCom(Port);

while ~KbCheck
    % first byte indicates type of message
    if T.bytesAvailable()      
        msgType = T.read(1);
        switch msgType
            case 1  % video to load
                if T.bytesAvailable() > 2
                    %[numFrames, videoIndex] = T.read(2);   % next two bytes are number of frames, and position index to load video                    
                    msg = T.read(2);   % next two bytes are number of frames, and position index to load video  
                    numFrames = msg(1);
                    videoIndex = msg(2);
                    if T.bytesAvailable() >= numFrames
                        video = T.read(numFrames);  % download video with indicated number of frames
                        BpodSystem.PluginObjects.V.Videos{videoIndex} = struct;
                        BpodSystem.PluginObjects.V.Videos{videoIndex}.nFrames = numFrames;
                        BpodSystem.PluginObjects.V.Videos{videoIndex}.Data = double(video); % convert unit8 to double
                        %BpodSystem.PluginObjects.V.loadVideo(videoIndex, video);
                    end
                end
            case 2  % play/stop command issued
                if T.bytesAvailable()
                    command = T.read(1);    % read command
                    switch command
                        case 1  % command is play
                            if T.bytesAvailable()
                                videoIndex = T.read(1); % get video index
                                %BpodSystem.PluginObjects.V.play(videoIndex);

                                
                                % Here we now specify a time at which PTB should be ready to draw to the
                                % screen by. In this example we use half a inter-frame interval. This
                                % specification allows us to get an accurate idea of whether PTB is making
                                % the stimulus timings we want.
                                vbl = Screen('Flip', BpodSystem.PluginObjects.V.Window);

                                for iFrame = 1:BpodSystem.PluginObjects.V.Videos{videoIndex}.nFrames
                                    if T.bytesAvailable()   % only command during video play is stop
                                        command = T.read(1);
                                        break;
                                    end
                                    Screen('DrawTexture', BpodSystem.PluginObjects.V.Window, BpodSystem.PluginObjects.V.Videos{videoIndex}.Data(iFrame));
                                    %[VBLts, SonsetTime, FlipTimestamp, Missed, BeamPos] = Screen('Flip', BpodSystem.PluginObjects.V.Window);
                                    % Flip to the screen
                                    vbl = Screen('Flip', BpodSystem.PluginObjects.V.Window, vbl + (waitframes - 0.5) * BpodSystem.PluginObjects.V.ifi);
                                end
                            end
                        case 2
                            %BpodSystem.PluginObjects.V. % code to stop
                            %video
                    end
                end

        end


        % [myInt16, myDouble] = T.read(10, 'int16', 50, 'double');
        disp(['msg:', num2str(msg)]);
    end

    % switch msg
    %     case 1
    %         BpodSystem.PluginObjects.V.play(5);
    %     otherwise
    % 
    % end
end

clear T;