function passive_temporal_sequence_hilbert
    global BpodSystem
    MonitorID  = 2;
    

    %% Import scripts

    m_InitGUI      = InitGUI;
    m_TrialConfig  = TrialConfig;
    m_AVstimConfig = AVstimConfig;
    m_OptoConfig   = OptoConfig;
    
    
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
    disp('OptoMode 1: off')
    disp('OptoMode 2: on')
    disp('OptoMode 3: default')
    disp('OptoMode 4: block')
    disp('OptoMode 5: Random')
    OptoMode = input('Input number to set opto mode >> ');
    
    global S
    [S] = m_InitGUI.SetParams(BpodSystem);
    [S] = m_InitGUI.UpdateJitter(S, FixJitterMode);
    [S] = m_InitGUI.UpdateNormalMode(S, NormalMode);
    [S] = m_InitGUI.UpdateOddballMode(S, OddballMode);
    [S] = m_InitGUI.UpdateOpto(S, OptoMode);

    BpodParameterGUI('init', S);


    %% Define stimuli and send to analog module
    
    SF = 192000; % Use lower sampling rate (samples/sec) to allow for longer duration audio file (max length limited by HiFi)
    H.SamplingRate = SF;
    Envelope = 1/(SF*0.001):1/(SF*0.001):1; % Define 1ms linear ramp envelope of amplitude coefficients, to apply at sound onset + in reverse at sound offset
    H.DigitalAttenuation_dB = -35; % Set a comfortable listening level for most headphones (useful during protocol dev).


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
    VisStim.OddballFlag = 0;
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
    % 1: opto on for oddball
    % 2: opto on for post oddball
    % 3: opto on for normal
    [OptoTypes] = m_TrialConfig.GenOptoTypes(S, TrialTypes);

    % isi
    [ISIseq] = m_AVstimConfig.GetISIseq( ...
        S, TrialTypes, NormalTypes, FixJitterTypes, OddballTypes);

    % video sequence
    [VisStimSeq] = m_AVstimConfig.GetVisStimSeq( ...
        S, BpodSystem, FPS, VisStim, TrialTypes, ISIseq);
    % audio sequence
    [AudStimSeq] = m_AVstimConfig.GetAudStimSeq( ...
        S, SF, Envelope, TrialTypes, ISIseq);
    % opto sequence
    [OptoSeq] = m_OptoConfig.GetOptoSeq(S, TrialTypes, OptoTypes, ISIseq);
    
    % save into session data
    BpodSystem.Data.ImgSeqLabel = ImgSeqLabel(1:S.GUI.MaxImg);
    BpodSystem.Data.NormalTypes = NormalTypes(1:S.GUI.MaxImg);
    BpodSystem.Data.FixJitterTypes = FixJitterTypes(1:S.GUI.MaxImg);
    BpodSystem.Data.OddballTypes = OddballTypes(1:S.GUI.MaxImg);
    BpodSystem.Data.OptoTypes = OptoTypes(1:S.GUI.MaxImg);
    BpodSystem.Data.ISIseq = ISIseq;
    SaveBpodSessionData


    %% Main trial loop

    for currentTrial = 1:length(VisStimSeq)
        S = BpodParameterGUI('sync', S);

        if (currentTrial == 1)
            pause(S.GUI.SpontSilenceTimeSess)
        end
        sma = NewStateMatrix();
        

        %% construct vis stim videos and audio stim
        
        VisStim = VisStimSeq(currentTrial);
        AudStim = AudStimSeq(currentTrial);

        BpodSystem.PluginObjects.V.Videos{25} = struct;
        BpodSystem.PluginObjects.V.Videos{25}.nFrames = length(VisStim.Data.Full); 
        BpodSystem.PluginObjects.V.Videos{25}.Data = VisStim.Data.Full;
        H.load(5, AudStim.Data);


        %% construct state matrix

        % opto and shutter control
        [sma] = m_OptoConfig.SetOpto(S, sma, OptoSeq, currentTrial);

        if OptoTypes(currentTrial)
            AudVisStimPlay_OutputActions = {'HiFi1', ['P', 4], 'GlobalTimerTrig', '011'};
        else
            AudVisStimPlay_OutputActions = {'HiFi1', ['P', 4]};
        end

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
