function omission_detection

    global BpodSystem
    global M
    
    EnableMovingSpouts = 0;
    MonitorID          = 2;
    EnableOpto         = 0;

    
    %% Import scripts
    
    m_Plotter      = Plotter;
    m_InitGUI      = InitGUI;
    m_TrialConfig  = TrialConfig;
    m_AVstimConfig = AVstimConfig;
    m_PostProcess  = PostProcess;
    m_Opto = OptoConfig(EnableOpto);
    
    
    %% Turn off Bpod LEDs
    
    BpodSystem.setStatusLED(0);

    
    %% Assert HiFi module is present + USB-paired (via USB button on console GUI)
    
    BpodSystem.assertModule('HiFi', 1);
    H = BpodHiFi(BpodSystem.ModuleUSB.HiFi1);
    
    
    %% Define parameters
    
    global S
    
    [S] = m_InitGUI.SetParams(BpodSystem);
    [S] = m_InitGUI.UpdateMovingSpouts(S, EnableMovingSpouts);
    
    
    %% Connect Maestro
    
    if (EnableMovingSpouts == 1)
        M = PololuMaestro('COM13');
        M.setMotor(0, m_TrialConfig.ConvertMaestroPos(S.GUI.RightServoInPos - S.GUI.ServoDeflection));
        M.setMotor(1, m_TrialConfig.ConvertMaestroPos(S.GUI.LeftServoInPos + S.GUI.ServoDeflection));
    end
    
    
    %% Trial initilization
    
    % set max number of trials
    
    BpodSystem.Data.TrialTypes = [];
    BpodSystem.Data.OptoTrialTypes = [];    % store opto trial types as 1-off, 2-on (could later generate as arrays of 0 & 1)
    BpodSystem.Data.OptoTag = [];    % store opto trial types as 1-off, 2-on
    BpodSystem.Data.ProcessedSessionData = {};

    % generate trial types
    [TrialTypes] = m_TrialConfig.GenTrials(S);
    
    % initial opto trial type generate (random)
    [OptoTrialTypes] = m_Opto.GenOptoTrials(BpodSystem, S);

    % Side Outcome Plot
    BpodSystem.ProtocolFigures.OutcomePlotFig = figure('Position', [50 540 1000 220],'name','Outcome plot','numbertitle','off', 'MenuBar', 'none', 'Resize', 'off');
    BpodSystem.GUIHandles.OutcomePlot = axes('Position', [.075 .35 .89 .55]);
    TrialTypeOutcomePlot(BpodSystem.GUIHandles.OutcomePlot, 'init', TrialTypes);
    BpodParameterGUI('init', S); % Initialize parameter GUI plugin
    
    currentTrial = 1;
    [OptoTrialTypes] = m_Opto.UpdateOptoTrials(BpodSystem, S, OptoTrialTypes, currentTrial, 1);
    m_Plotter.UpdateOutcomePlot(BpodSystem, TrialTypes, OptoTrialTypes, 0);
    
    
    %% Define stimuli and send to analog module
    
    SF = 44100; % Use lower sampling rate (samples/sec) to allow for longer duration audio file (max length limited by HiFi)
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
    VisStim.OmiFlag = 'False';

    [VisStim] = m_AVstimConfig.GetVisStimImg(S, BpodSystem, FPS, VisStim);
    GrayInitBNCSync = [repmat(VisStim.Img.GrayFrame_SyncW, 1, 120) VisStim.Img.GrayFrame_SyncBlk];
    BpodSystem.PluginObjects.V.Videos{6} = struct;
    BpodSystem.PluginObjects.V.Videos{6}.nFrames = 121;
    BpodSystem.PluginObjects.V.Videos{6}.Data = GrayInitBNCSync;

    pause(1.0);
    BpodSystem.PluginObjects.V.TimerMode = 0;
    BpodSystem.PluginObjects.V.play(0);
    BpodSystem.SoftCodeHandlerFunction = 'SoftCodeHandler';    
    BpodSystem.PluginObjects.V.play(6); 
    input('Set parameters and press enter to continue >', 's'); 
    S = BpodParameterGUI('sync', S);
    
    
    %% Main trial loop
    
    for currentTrial = 1:S.GUI.MaxTrials
    
        S = BpodParameterGUI('sync', S);

        LeftValveTime   = m_TrialConfig.Amount2Time(3, 1);
        RightValveTime  = m_TrialConfig.Amount2Time(3, 3);

    
        %% duration related configs
    
        DURA.TimeOutPunish = m_TrialConfig.GetTimeOutPunish(S);
        DURA.ITI = m_TrialConfig.GetITI(S);
        DURA.ChoiceWindow = m_TrialConfig.GetChoiceWindow(S);


        %% update opto trial types and outcome plot
        % update for opto interop
        [OptoTrialTypes] = m_Opto.UpdateOptoTrials(BpodSystem, S, OptoTrialTypes, currentTrial, 0);
        if ~m_Opto.EnableOpto
            OptoStateExpInfo = 'Control';
            OptoTrialExpInfo = 'NA';
        else
            OptoStateExpInfo = 'Opto';
            switch OptoTrialTypes(currentTrial)
                case 1                    
                    OptoTrialExpInfo = 'Opto Off';
                case 2                    
                    OptoTrialExpInfo = 'Opto On';
            end
        end

        switch OptoTrialTypes(currentTrial)
            case 1
                BpodSystem.Data.OptoTag(currentTrial) = 0;
            case 2
                BpodSystem.Data.OptoTag(currentTrial) = 1;
        end

        m_Plotter.UpdateOutcomePlot(BpodSystem, TrialTypes, OptoTrialTypes, 0);

        %% construct preperturb vis stim videos and audio stim base for grating and gray if duration parameters changed
        
        % config audio stimulus
        m_AVstimConfig.ConfigHifi(H, S, SF, Envelope);
    
        [JitterFlag] = m_TrialConfig.GetJitterFlag(S);
    
        % generate video
        VisStim.ProcessedData.Seq = [];
        [VisStim] = m_AVstimConfig.GetVisStimImg(S, BpodSystem, FPS, VisStim);

        VisStim = m_AVstimConfig.GetVideoData(S, BpodSystem, FPS, VisStim, JitterFlag, TrialTypes(currentTrial));
    
        % combine full video
        VisStim.Data.Full = m_AVstimConfig.GetFullVideo(S, VisStim, FPS);
    
        % load constructed video into the video object
        BpodSystem.PluginObjects.V.Videos{25} = struct;
        BpodSystem.PluginObjects.V.Videos{25}.nFrames = length(VisStim.Data.Full); 
        BpodSystem.PluginObjects.V.Videos{25}.Data = VisStim.Data.Full;

        VisStimDuration = 3;

    
        %% Generate audio stim based on vis stim for this trial
        
        m_AVstimConfig.ConfigFullAudioStim( ...
            H, S, VisStim, SF, Envelope, TrialTypes(currentTrial));

    
        %% set softcode for hardware control output actions
    
        SCOA.Start       = {'HiFi1','*', 'BNC1', 1};
        SCOA.InitCue     = {'HiFi1', ['P' 0]};
        SCOA.StimAct     = m_AVstimConfig.GetStimAct(S, m_Opto.EnableOpto);
        SCOA.EarlyChoice = {'SoftCode', 255, 'HiFi1', 'X'};
        SCOA.Punish      = {'SoftCode', 255, 'HiFi1', ['P' 2]};
        if (S.GUI.VisStimEnable == 1)
            SCOA.VisStim = {'SoftCode', 25};
        else
            SCOA.VisStim = {};
        end
        SCOA.AudStim     = m_Opto.GetAudStimOpto(OptoTrialTypes(currentTrial));

        if (EnableMovingSpouts == 1)
            SCOA.SpoutIn = {'SoftCode', 9};
        else
            SCOA.SpoutIn = {};
        end

    
        %% construct state matrix

        sma = NewStateMatrix();

        switch S.GUI.TrainingLevel
            case 1 % naive
                ExperimenterTrialInfo.TrainingLevel = 'Naive';
                StateNaive(sma, S, SCOA, VisStimDuration, DURA);
        end
    
    
        %% add console print for experimenter trial information
    
        ExperimenterTrialInfo.TrialNumber = currentTrial;
    
        strExperimenterTrialInfo = formattedDisplayText(ExperimenterTrialInfo,'UseTrueFalseForLogical',true);
        disp(strExperimenterTrialInfo);
    
        RawEvents = RunStateMachine;
        
    
        %% save data and update plot
    
        if ~isempty(fieldnames(RawEvents))
            BpodSystem.Data = AddTrialEvents(BpodSystem.Data,RawEvents);
            BpodSystem.Data.TrialSettings(currentTrial) = S;
            BpodSystem.Data.TrialTypes(currentTrial) = TrialTypes(currentTrial);
            BpodSystem.Data.JitterFlag(currentTrial) = JitterFlag;
            BpodSystem.Data.OptoTrialTypes(currentTrial) = OptoTrialTypes(currentTrial);
            m_PostProcess.SaveProcessedSessionData(BpodSystem, VisStim);
            m_Plotter.UpdateOutcomePlot(BpodSystem, TrialTypes, OptoTrialTypes, 1);
            StateTiming();
            SaveBpodSessionData;
        end
        HandlePauseCondition;
        if BpodSystem.Status.BeingUsed == 0
            clear global M;
            BpodSystem.PluginObjects.V = [];
            BpodSystem.setStatusLED(1);
            return
        end
    
    end
    
    clear global M;
    BpodSystem.PluginObjects.V = [];
    BpodSystem.setStatusLED(1);
