function passive_temporal_sequence_hilbert_stamm_opto_V_1_2
    global BpodSystem
    MonitorID  = 2;
    

    %% Import scripts

    m_InitGUI      = InitGUI;
    m_TrialConfig  = TrialConfig;
    m_AVstimConfig = AVstimConfig;
    % m_Opto = OptoConfig;
    
    
    %% Turn off Bpod LEDs
    
    BpodSystem.setStatusLED(0);
    
    %% Assert HiFi module is present + USB-paired (via USB button on console GUI)
    
    BpodSystem.assertModule('HiFi', 1);
    H = BpodHiFi(BpodSystem.ModuleUSB.HiFi1);
    
    
    %% Define parameters
    
    disp('FixJitterMode 1: fix')
    disp('FixJitterMode 2: jitter')
    disp('FixJitterMode 3: random')
    disp('FixJitterMode 4: block')
    FixJitterMode = input('Input number to set fix or jitter >> ');
    disp('NormalMode 1: short')
    disp('NormalMode 2: long')
    disp('NormalMode 3: random')
    disp('NormalMode 4: block')
    NormalMode = input('Input number to set normal mode >> ');
    disp('OddballMode 1: short')
    disp('OddballMode 2: long')
    disp('OddballMode 3: random')
    disp('OddballMode 4: block')
    OddballMode = input('Input number to set oddball mode >> ');
    disp('EnableOpto 0: disable')
    disp('EnableOpto 1: enable')
    EnableOpto = input('Input number to set opto mode >> ');
    
    global S
    [S] = m_InitGUI.SetParams(BpodSystem);
    [S] = m_InitGUI.UpdateJitter(S, FixJitterMode);
    [S] = m_InitGUI.UpdateNormalMode(S, NormalMode);
    [S] = m_InitGUI.UpdateOddballMode(S, OddballMode);
    [S] = m_InitGUI.UpdateOpto(S, EnableOpto);

    % BpodSystem.Data.OptoTag = [];    % store opto trial types as 1-off, 2-on
    % [OptoTrialTypes] = m_Opto.GenOptoTrials(BpodSystem, S);
    % [OptoTrialTypes] = m_Opto.UpdateOptoTrials(BpodSystem, S, OptoTrialTypes, 1, 1);

    BpodParameterGUI('init', S);


    %% Define stimuli and send to analog module
    
    SF = 192000; % Use lower sampling rate (samples/sec) to allow for longer duration audio file (max length limited by HiFi)
    H.SamplingRate = SF;
    Envelope = 1/(SF*0.001):1/(SF*0.001):1; % Define 1ms linear ramp envelope of amplitude coefficients, to apply at sound onset + in reverse at sound offset
    
    H.DigitalAttenuation_dB = -35; % Set a comfortable listening level for most headphones (useful during protocol dev).

    m_AVstimConfig.ConfigFullAudioStim( ...
        H, S, SF, Envelope);
    

    %% Setup video
    
    BpodSystem.PluginObjects.V = [];
    BpodSystem.PluginObjects.V = PsychToolboxVideoPlayer(MonitorID, 0, [0 0], [180 180], 0); % Assumes Sync patch = 180x180 pixels
    BpodSystem.PluginObjects.V.SyncPatchIntensity = 255;
    BpodSystem.PluginObjects.V.loadText(1, 'Loading...', '', 80);
    BpodSystem.PluginObjects.V.play(1);
    Xsize = BpodSystem.PluginObjects.V.ViewPortDimensions(1);
    Ysize = BpodSystem.PluginObjects.V.ViewPortDimensions(2);
    FPS   = BpodSystem.PluginObjects.V.DetectedFrameRate;
    
    % load frame images
    
    [VideoGrayFixed] = m_AVstimConfig.GenGreyImg(Xsize, Ysize);
    BpodSystem.PluginObjects.V.loadVideo(1, VideoGrayFixed);
    
    ImgParams.spatialFreq = 0.005;
    ImgParams.contrast    = 1;
    ImgParams.phase       = 0.5;
    
    ImgParams.orientation = 0;
    [VideoGrating] = m_AVstimConfig.GenStimImg(ImgParams, Xsize, Ysize);
    BpodSystem.PluginObjects.V.loadVideo(2, VideoGrating);
    
    ImgParams.orientation = 45;
    [VideoGrating] = m_AVstimConfig.GenStimImg(ImgParams, Xsize, Ysize);
    BpodSystem.PluginObjects.V.loadVideo(3, VideoGrating);
    
    ImgParams.orientation = 90;
    [VideoGrating] = m_AVstimConfig.GenStimImg(ImgParams, Xsize, Ysize);
    BpodSystem.PluginObjects.V.loadVideo(4, VideoGrating);
    
    ImgParams.orientation = 135;
    [VideoGrating] = m_AVstimConfig.GenStimImg(ImgParams, Xsize, Ysize);
    BpodSystem.PluginObjects.V.loadVideo(5, VideoGrating);
    
    VisStim.orientation = [0 45 90 135];
    VisStim.GratingIdx = [2 3 4 5];
    VisStim.OddballFlag = 'False';
    VisStim.OddballISI = 0.19961106;

    [VisStim] = m_AVstimConfig.GetVisStimImg(S, BpodSystem, FPS, VisStim, 2);
    GrayInitBNCSync = [repmat(VisStim.Img.GrayFrame_SyncW, 1, 120) VisStim.Img.GrayFrame_SyncBlk];
    BpodSystem.PluginObjects.V.Videos{6} = struct;
    BpodSystem.PluginObjects.V.Videos{6}.nFrames = 121;
    BpodSystem.PluginObjects.V.Videos{6}.Data = GrayInitBNCSync;

    pause(1.0);
    BpodSystem.PluginObjects.V.TimerMode = 1;
    BpodSystem.PluginObjects.V.play(0);
    BpodSystem.SoftCodeHandlerFunction = 'SoftCodeHandler';    
    BpodSystem.PluginObjects.V.play(6);

    input('Set parameters and press enter to continue >> ', 's'); 
    S = BpodParameterGUI('sync', S);

    %% sequence configuration and visualization

    % trial type
    % 1: oddball
    % 2: 0 orien stim
    % 3: 45 orien stim
    % 4: 90 orien stim
    % 5: 135 orien stim
    [TrialTypes, ImgSeqLabel] = m_TrialConfig.GenTrialTypesSeq(S);
    
    % baseline type
    % 0: short ISI
    % 1: long ISI
    [NormalTypes] = m_TrialConfig.GenNormalTypes(S);
    
    % fix jitter type
    % 0: fix
    % 1: jitter
    [FixJitterTypes] = m_TrialConfig.GenFixJitterTypes(S);
    
    % oddball type
    % 0: short oddball
    % 1: long oddball
    [OddballTypes] = m_TrialConfig.GenOddballTypes(S);

    % opto type
    % 0: opto off
    % 1: opto on
    [OptoTypes] = m_TrialConfig.GenOptoTypes(S);

    % isi
    [ISIseq] = m_AVstimConfig.GetISIseq( ...
        S, TrialTypes, NormalTypes, FixJitterTypes, OddballTypes);
    
    % save into session data
    BpodSystem.Data.ImgSeqLabel = ImgSeqLabel(1:S.GUI.MaxImg);
    BpodSystem.Data.NormalTypes = NormalTypes(1:S.GUI.MaxImg);
    BpodSystem.Data.FixJitterTypes = FixJitterTypes(1:S.GUI.MaxImg);
    BpodSystem.Data.OddballTypes = OddballTypes(1:S.GUI.MaxImg);
    BpodSystem.Data.OptoTypes = OptoTypes(1:S.GUI.MaxImg);
    BpodSystem.Data.ISIseq = ISIseq;
    SaveBpodSessionData


    %% Main trial loop

    for currentTrial = 1:19961106
        S = BpodParameterGUI('sync', S);

        if currentTrial > S.GUI.MaxImg
            break;
        end

        if (currentTrial == 1)
            pause(S.GUI.SpontSilenceTimeSess)
        end
        
        
        %% construct preperturb vis stim videos and audio stim base for grating and gray if duration parameters changed
        
        % generate video
        VisStim = m_AVstimConfig.GetVideoData( ...
            S, BpodSystem, FPS, VisStim, ...
            TrialTypes(currentTrial), ...
            ISIseq(currentTrial), OptoTypes(currentTrial));

        % load constructed video into the video object
        BpodSystem.PluginObjects.V.Videos{25} = struct;
        BpodSystem.PluginObjects.V.Videos{25}.nFrames = length(VisStim.Data.Full); 
        BpodSystem.PluginObjects.V.Videos{25}.Data = VisStim.Data.Full;

        %% create state matrix

        sma = NewStateMatrix();

        %% add opto to constructed video/audio stim base

        if OptoTypes(currentTrial)
            % pmt shutter delay timings are from Bruker specs, the actual
            % timings are a bit different, and vary between green/red PMTs, but these work for pulsing opto            

            % shutter close delays
            PMTStartCloseDelay = 0.010;     % ~10ms for shutter to start closing after setting shutter signal to 5V
            PMTCloseTransferDelay = 0.0121; % ~12.1ms
            PMTCloseDelay = PMTStartCloseDelay + PMTCloseTransferDelay;          
            PMTMin5VSignalDur = PMTCloseDelay + 0.003; % ~25.1ms, minimum measured duration of 5V shutter signal for shutter to re-open 
                       
            % shutter open delays
            PMTStartOpenDelay = 0.0078;     % ~7.8ms for shutter to start opening after setting shutter signal to 0V
            PMTOpenTransferDelay = 0.0125;  % ~12.5ms for shutter to open after the 7.8ms open transfer delay
            
            % 2p scope image dur for 30Hz Resonant Galvo
            % ScopeFrameDuration = 0.033;     % ~33ms, in case need to make
            % any calculations to adjust for scope image duration
                      
            % opto cycle timing is defined by the onset of the LED
            LEDOnsetDelay = ISIseq(currentTrial) - S.GUI.OptoPreVisOnset; 
            PMTOnsetDelay = LEDOnsetDelay - PMTCloseDelay;

            % 10Hz pulsed shutter/opto, 7.8ms LED
            LEDOnPulseDur = S.GUI.LEDOnPulseDur;
            LEDOffDur = S.GUI.OptoFreq - LEDOnPulseDur;
          
            % PMT shutter signal 5V and 0V durations           
            PMT5VDur = PMTCloseDelay; % set PMT5V dur initially to shutter close delay
            
            % if the LED is on for longer than the shutter StartOpenDelay,
            % then increase shutter 5V duration by the difference (LEDOnPulseDur - PMTStartOpenDelay)
            if LEDOnPulseDur > PMTStartOpenDelay
                PMT5VDur = PMT5VDur + (LEDOnPulseDur - PMTStartOpenDelay);
            end

            % if shutter duration is less than the minimum dur for the
            % shutter to re-open, set it to minimum shutter pulse dur
            PMT5VDur = max(PMT5VDur, PMTMin5VSignalDur);

            % PMT0VDur = LEDOnPulseDur + PMTOpenTransferDelay + 2*ScopeFrameDuration - PMTStartCloseDelay - ShutterPulseWidthAdd;
            % duration of LED pulse 0V is cycle period minus 5V dur
            PMT0VDur =  S.GUI.OptoFreq - PMT5VDur;

            % integer number of pmt/led cycles within pre + post grating onset
            numPMTLEDCycles = floor((S.GUI.OptoPreVisOnset + S.GUI.OptoPostVisOnset) / S.GUI.OptoFreq);

            LoopLED = numPMTLEDCycles;
            LoopPMT = numPMTLEDCycles;
            
            % LED timers
            sma = SetGlobalTimer(sma, 'TimerID', 1, 'Duration', LEDOnPulseDur, 'OnsetDelay', LEDOnsetDelay,...
                'Channel', 'PWM1', 'OnLevel', 255, 'OffLevel', 0,...
                'Loop', LoopLED, 'SendGlobalTimerEvents', 0, 'LoopInterval', LEDOffDur,...
                'GlobalTimerEvents', 0, 'OffsetValue', 0);

            % PMT shutter timers
            sma = SetGlobalTimer(sma, 'TimerID', 2, 'Duration', PMT5VDur, 'OnsetDelay', PMTOnsetDelay,...
                'Channel', 'BNC2', 'OnLevel', 1, 'OffLevel', 0,...
                'Loop', LoopPMT, 'SendGlobalTimerEvents', 0, 'LoopInterval', PMT0VDur,...
                'GlobalTimerEvents', 0, 'OffsetValue', 0);                 
        end

        %% add opto timer triggers
        AudVisStimPlay_OutputActions = {'HiFi1', ['P', 4]};

        if OptoTypes(currentTrial)
            AudVisStimPlay_OutputActions = [AudVisStimPlay_OutputActions, {'GlobalTimerTrig', '011'}];
        end

        %% construct state matrix

        % sma create above to set timers
    
        sma = AddState(sma, 'Name', 'Start', ...
            'Timer', 0,...
            'StateChangeConditions', {'Tup', 'VisStimTrigger'},...
            'OutputActions', {'HiFi1','*'});
    
        sma = AddState(sma, 'Name', 'VisStimTrigger', ...
            'Timer', 0,...
            'StateChangeConditions', {'BNC1High', 'AudVisStimPlay'},...
            'OutputActions', {'SoftCode', 25});

        sma = AddState(sma, 'Name', 'AudVisStimPlay', ...
            'Timer', VisStim.Data.Dur,...
            'StateChangeConditions', {'Tup', 'End'},...
            'OutputActions', AudVisStimPlay_OutputActions);
        
        sma = AddState(sma, 'Name', 'End', ...
            'Timer', 0,...
            'StateChangeConditions', {'Tup', '>exit'},...
            'OutputActions', {'SoftCode', 254, 'HiFi1', 'X'});
        
        SendStateMachine(sma);

        RunStateMachine;


        %% add console print for experimenter trial information

        m_TrialConfig.DispExpInfo( ...
            ImgSeqLabel, NormalTypes, FixJitterTypes, OddballTypes, ...
            currentTrial, VisStim)


        %% save data

        HandlePauseCondition;
        if BpodSystem.Status.BeingUsed == 0
            clear global M;
            BpodSystem.PluginObjects.V = [];
            BpodSystem.setStatusLED(1);
            return
        end


    end

    input('Session successfully ended. press enter to exit >> ', 's'); 
    
    clear global M;
    BpodSystem.PluginObjects.V = [];
    BpodSystem.setStatusLED(1);
