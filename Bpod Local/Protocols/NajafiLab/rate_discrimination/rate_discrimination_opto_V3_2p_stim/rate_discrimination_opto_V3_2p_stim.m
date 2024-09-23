function rate_discrimination_opto_V3_2p_stim

    global BpodSystem
    global M
    MonitorID = 2;

    disp('0: off')
    disp('1: on')
    EnableMovingSpouts = input('Set moving spout >> ');
    disp('Opto fraction')
    OnFraction = input('Set opto trial fraction >> ');
    

    %% Import scripts
    
    m_Plotter      = Plotter;
    m_InitGUI      = InitGUI;
    m_TrialConfig  = TrialConfig;
    m_AVstimConfig = AVstimConfig;
    m_PostProcess  = PostProcess;
    m_Opto         = OptoConfig;
    
    
    %% Turn off Bpod LEDs
    
    BpodSystem.setStatusLED(0);


    %% Assert HiFi module is present + USB-paired (via USB button on console GUI)
    
    BpodSystem.assertModule('HiFi', 1);
    H = BpodHiFi(BpodSystem.ModuleUSB.HiFi1);
    
    
    %% Define parameters
    
    global S
    
    [S] = m_InitGUI.SetParams(BpodSystem);
    [S] = m_InitGUI.UpdateMovingSpouts(S, EnableMovingSpouts);
    [S] = m_InitGUI.UpdateOpto(S, OnFraction);
    
    
    %% Connect Maestro
    
    if (EnableMovingSpouts == 1)
        M = PololuMaestro('COM15');
        M.setMotor(0, m_TrialConfig.ConvertMaestroPos(S.GUI.RightServoInPos - S.GUI.ServoDeflection));
        M.setMotor(1, m_TrialConfig.ConvertMaestroPos(S.GUI.LeftServoInPos + S.GUI.ServoDeflection));
    end
    
    
    %% Trial initilization
    
    % set max number of trials
    
    BpodSystem.Data.TrialTypes = [];
    BpodSystem.Data.OptoType = [];
    BpodSystem.Data.ProcessedSessionData = {};
    
    % initialize anti-bias variables
    AntiBiasVar.IncorrectFlag       = 0;
    AntiBiasVar.IncorrectType       = 1;
    AntiBiasVar.CompletedHist.left  = [];
    AntiBiasVar.CompletedHist.right = [];
    AntiBiasVar.BiasIndex           = 0;
    AntiBiasVar.ValveFlag           = 'NoBias';
    TotalRewardAmount_uL            = 0;
    
    % draw perturbation interval from uniform distribution in range
    PerturbInterval.EasyMinPercent       = 3/4;
    PerturbInterval.EasyMaxPercent       = 1;
    PerturbInterval.MediumEasyMinPercent = 1/2;
    PerturbInterval.MediumEasyMaxPercent = 3/4;
    PerturbInterval.MediumHardMinPercent = 1/4;
    PerturbInterval.MediumHardMaxPercent = 1/2;
    PerturbInterval.HardMinPercent       = 0;
    PerturbInterval.HardMaxPercent       = 1/4;
    
    % generate trial types
    [TrialTypes] = m_TrialConfig.GenTrials(S);
    [TrialTypes] = m_TrialConfig.AdjustWarmupTrials(S, TrialTypes);
    [OptoType]   = m_Opto.GenOptoType(S);

    % Side Outcome Plot
    BpodSystem.ProtocolFigures.OutcomePlotFig = figure('Position', [50 540 1000 220],'name','Outcome plot','numbertitle','off', 'MenuBar', 'none', 'Resize', 'off');
    BpodSystem.GUIHandles.OutcomePlot = axes('Position', [.075 .35 .89 .55]);
    TrialTypeOutcomePlot(BpodSystem.GUIHandles.OutcomePlot, 'init', TrialTypes);
    BpodParameterGUI('init', S); % Initialize parameter GUI plugin
    m_Plotter.UpdateOutcomePlot(BpodSystem, TrialTypes, 0);
    
    
    %% Define stimuli and send to analog module
    
    SF = 44100;
    H.SamplingRate = SF;
    Envelope = 1/(SF*0.001):1/(SF*0.001):1;
    H.DigitalAttenuation_dB = -35;
    
    
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
    BpodSystem.PluginObjects.V.TimerMode = 2;
    BpodSystem.PluginObjects.V.play(0);
    BpodSystem.SoftCodeHandlerFunction = 'SoftCodeHandler';    
    BpodSystem.PluginObjects.V.play(6); 
    input('Set parameters and press enter to continue >', 's'); 
    S = BpodParameterGUI('sync', S);
    
    
    %% Main trial loop
    
    for currentTrial = 1:S.GUI.MaxTrials
    
        S = BpodParameterGUI('sync', S);
    
        %% anti bias
    
        [TrialTypes] = m_TrialConfig.ManuallFraction( ...
            S, currentTrial, TrialTypes);

        [AntiBiasVar, LeftValveAmount_uL, RightValveAmount_uL, TrialTypes] = m_TrialConfig.AntiBiasValveAdjust( ...
            BpodSystem, S, AntiBiasVar, currentTrial, TrialTypes);
    
        LeftValveTime   = m_TrialConfig.Amount2Time(LeftValveAmount_uL, 1);
        RightValveTime  = m_TrialConfig.Amount2Time(RightValveAmount_uL, 3);
    
    
        %% get difficulty level for this trial
        
        [TrialDifficulty] = m_TrialConfig.SamplingDiff(S);
    
        switch TrialDifficulty
            case 1
                ExperimenterTrialInfo.Difficulty = 'Easy';
            case 2
                ExperimenterTrialInfo.Difficulty = 'Medium-Easy';
            case 3
                ExperimenterTrialInfo.Difficulty = 'Medium-Hard';
            case 4
                ExperimenterTrialInfo.Difficulty = 'Hard';
        end
    
        %% duration related configs
    
        DURA.TimeOutPunish = m_TrialConfig.GetTimeOutPunish(S);
        DURA.ITI = m_TrialConfig.GetITI(S);
        DURA.ChoiceWindow = m_TrialConfig.GetChoiceWindow(S);
        DURA.ChangeMindDur = m_TrialConfig.GetChangeMindDur(S);
        
    
        %% set vis stim perturbation ISI duration according to trial-specific difficulty level
    
        % perturbation sampling
        [PostPertISI, EasyMaxInfo] = m_TrialConfig.GetPostPertISI( ...
                S, TrialDifficulty, PerturbInterval);
        [GrayPerturbISI] = m_TrialConfig.SetPostPertISI( ...
            S, TrialTypes, currentTrial, PostPertISI);
    
        BpodSystem.Data.TrialVars.Trial{currentTrial}.PostPertISI = PostPertISI + S.GUI.GratingDur_s;


        %% construct preperturb vis stim videos and audio stim base for grating and gray if duration parameters changed
        
        % config audio stimulus
        m_AVstimConfig.ConfigHifi(H, S, SF, Envelope);
    
        [JitterFlag] = m_TrialConfig.GetJitterFlag(S);
    
        % generate video
        VisStim.ProcessedData.Seq = [];
        VisStim.ProcessedData.PrePost = [];
        [VisStim] = m_AVstimConfig.GetVisStimImg(S, BpodSystem, FPS, VisStim);

        VisStim = m_AVstimConfig.GetVideoDataPre(S, BpodSystem, FPS, VisStim, JitterFlag);
        VisStim = m_AVstimConfig.GetVideoDataPost(S, BpodSystem, FPS, VisStim, GrayPerturbISI, JitterFlag);
        VisStim = m_AVstimConfig.GetVideoDataExtra(S, BpodSystem, FPS, VisStim, GrayPerturbISI, JitterFlag);
    
        % combine full video
        VisStim.Data.Full = m_AVstimConfig.GetFullVideo(S, VisStim, FPS);
    
        % add gray frames for opto
        OptoGrayFrames = [m_AVstimConfig.GetUnitVideo(VisStim.Img.GrayFrame_SyncW, 1) m_AVstimConfig.GetUnitVideo(VisStim.Img.GrayFrame_SyncBlk, 1)];
        VisStim.Data.OptoGrayDur = m_AVstimConfig.GetVideoDur(FPS, OptoGrayFrames);
        VisStim.Data.Full = [OptoGrayFrames VisStim.Data.Full];
       
        % load constructed video into the video object
        BpodSystem.PluginObjects.V.Videos{25} = struct;
        BpodSystem.PluginObjects.V.Videos{25}.nFrames = length(VisStim.Data.Full); 
        BpodSystem.PluginObjects.V.Videos{25}.Data = VisStim.Data.Full;

        % use vis stim duration for opto timing
        if (S.GUI.ReactionTask == 1)
            VisStim.Data.VisStimDuration = VisStim.Data.Pre.Dur;
        else
            VisStim.Data.VisStimDuration = VisStim.Data.Pre.Dur + VisStim.Data.Post.Dur;
        end


        %% Generate audio stim based on vis stim for this trial, account for shift due to gray frames
        
        [FullAudio] = m_AVstimConfig.GenAudioStim( ...
            S, VisStim, SF, Envelope);
        H.load(5, FullAudio);

        
        %% trial target
        % port1:left, port2:center, port3:right
        
        TargetConfig.Left.CorrectChoice = 'Left';
        TargetConfig.Left.CorrectLick   = 'Port1In';
        TargetConfig.Left.CorrectPort   = 'Port1';
        TargetConfig.Left.IncorrectLick = 'Port3In';
        TargetConfig.Left.IncorrectPort = 'Port3';
        TargetConfig.Left.Valve         = 'Valve1';
        TargetConfig.Left.ValveTime     = LeftValveTime;
        TargetConfig.Right.CorrectChoice = 'Right';
        TargetConfig.Right.CorrectLick   = 'Port3In';
        TargetConfig.Right.CorrectPort   = 'Port3';
        TargetConfig.Right.IncorrectLick = 'Port1In';
        TargetConfig.Right.IncorrectPort = 'Port1';
        TargetConfig.Right.Valve         = 'Valve3';
        TargetConfig.Right.ValveTime     = RightValveTime;
        switch TrialTypes(currentTrial)
            case 1
                TrialTarget = TargetConfig.Left;
            case 2
                TrialTarget = TargetConfig.Right;
        end
    
    
        %% set softcode for hardware control output actions

        SCOA.Start       = {'HiFi1','*', 'BNC1', 1};
        SCOA.InitCue     = {'HiFi1', ['P' 0]};
        SCOA.StimAct     = {};
        SCOA.Punish      = {'HiFi1', ['P' 2]};
        if (S.GUI.VisStimEnable == 1)
            SCOA.VisStim = {'SoftCode', 25};
        else
            SCOA.VisStim = {};
        end
        SCOA.AudStim     = m_Opto.OptoUpdateAudStimTrig(S, OptoType, currentTrial);
        SCOA.ResetTimer  = m_Opto.OptoUpdateResetTimer(S, OptoType, currentTrial);
        if (EnableMovingSpouts == 1)
            SCOA.SpoutIn = {'SoftCode', 9};
        else
            SCOA.SpoutIn = {};
        end

    
        %% construct state matrix

        sma = NewStateMatrix();
        sma = m_Opto.SetOpto(BpodSystem, S, sma, VisStim, OptoType, currentTrial);
        sma = m_Opto.Set2PStim(BpodSystem, S, sma, VisStim, OptoType, currentTrial);

        switch OptoType(currentTrial)
            case 0
                ExperimenterTrialInfo.Opto = 'off';
            case 1
                ExperimenterTrialInfo.Opto = 'on';
        end

        switch S.GUI.TrainingLevel
            case 1 % naive
                ExperimenterTrialInfo.TrainingLevel = 'Naive';
                StateNaive(sma, S, SCOA, TrialTarget, VisStim.Data.VisStimDuration, DURA);
            case 2 % Mid Trained
                if (currentTrial <= S.GUI.NumNaiveWarmup)
                    ExperimenterTrialInfo.TrainingLevel = 'Naive warmup';
                    StateNaive(sma, S, SCOA, TrialTarget, VisStim.Data.VisStimDuration, DURA);
                else
                    ExperimenterTrialInfo.TrainingLevel = 'Mid Trained';
                    StateMidTrain(sma, S, SCOA, TrialTarget, VisStim.Data.VisStimDuration, DURA);
                end
            case 3 % well trained
                if (currentTrial <= S.GUI.NumNaiveWarmup)
                    ExperimenterTrialInfo.TrainingLevel = 'Naive warmup';
                    StateNaive(sma, S, SCOA, TrialTarget, VisStim.Data.VisStimDuration, DURA);
                else
                    ExperimenterTrialInfo.TrainingLevel = 'Well Trained';
                    StateWellTrained(sma, S, SCOA, TrialTarget, VisStim.Data.VisStimDuration, DURA);
                end
        end
    
    
        %% add console print for experimenter trial information
        
        [TotalRewardAmount_uL] = m_TrialConfig.GetTotalRewardAmount( ...
            BpodSystem, S, TotalRewardAmount_uL, ...
            currentTrial, TrialTypes, LeftValveAmount_uL, RightValveAmount_uL);

        ExperimenterTrialInfo.TrialNumber = currentTrial;
        ExperimenterTrialInfo.GratingOrientation = VisStim.orientation(VisStim.SampleGratingIdx-1);
        ExperimenterTrialInfo.CorrectChoice = TrialTarget.CorrectChoice;
        ExperimenterTrialInfo.ReactionTask = S.GUI.ReactionTask;
        ExperimenterTrialInfo.Bias = AntiBiasVar.ValveFlag;
        ExperimenterTrialInfo.LeftValveAmount  = LeftValveAmount_uL;
        ExperimenterTrialInfo.RightValveAmount = RightValveAmount_uL;
        ExperimenterTrialInfo.PostPertISI = GrayPerturbISI;
        ExperimenterTrialInfo.VisStimDuration = VisStim.Data.Pre.Dur + VisStim.Data.Post.Dur + VisStim.Data.Extra.Dur;
        ExperimenterTrialInfo.CategoryBoundary = S.GUI.ISIOrig_s;
        ExperimenterTrialInfo.EasyMax = EasyMaxInfo;
        ExperimenterTrialInfo.ITI = DURA.ITI;
        ExperimenterTrialInfo.TimeOutPunish = DURA.TimeOutPunish;
        ExperimenterTrialInfo.ChoiceWindow = DURA.ChoiceWindow;
        ExperimenterTrialInfo.ChangeMindDur = DURA.ChangeMindDur;
        ExperimenterTrialInfo.PreISIinfo = VisStim.PreISIinfo;
        ExperimenterTrialInfo.PostISIinfo = VisStim.PostISIinfo;
        ExperimenterTrialInfo.Jitter = JitterFlag;
        ExperimenterTrialInfo.Omission = VisStim.OmiFlag;
        ExperimenterTrialInfo.TotalRewardAmount_uL = TotalRewardAmount_uL;
    
        strExperimenterTrialInfo = formattedDisplayText(ExperimenterTrialInfo,'UseTrueFalseForLogical',true);
        disp(strExperimenterTrialInfo);
    
        RawEvents = RunStateMachine;
        
    
        %% save data and update plot
    
        if ~isempty(fieldnames(RawEvents))
            BpodSystem.Data = AddTrialEvents(BpodSystem.Data,RawEvents);
            BpodSystem.Data.TrialSettings(currentTrial) = S;
            BpodSystem.Data.TrialTypes(currentTrial) = TrialTypes(currentTrial);
            BpodSystem.Data.OptoType(currentTrial) = OptoType(currentTrial);
            BpodSystem.Data.JitterFlag(currentTrial) = JitterFlag;
            m_PostProcess.SaveProcessedSessionData(BpodSystem, VisStim, GrayPerturbISI);
            m_Plotter.UpdateOutcomePlot(BpodSystem, TrialTypes, 1);
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
end
