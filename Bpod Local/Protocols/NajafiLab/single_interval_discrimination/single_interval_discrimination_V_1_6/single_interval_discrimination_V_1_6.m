function single_interval_discrimination_V_1_6
try
    % using try/catch for protocol code to allow for disconnecting from
    % modules and logging crash reports when exceptions thrown
    protocol_version = 'single_interval_discrimination_V_1_6';


    global BpodSystem
    global M
    global AntiBiasVar
    global TargetConfig
    
    % counter for plot filenames
    BpodSystem.Data.PlotCntr = 1;

    % current trial var for bpod global
    BpodSystem.Data.CurrentTrial = 1;

    % get rig ID from computer host name
    SetRigID(BpodSystem)    

    switch BpodSystem.Data.RigName
        case 'ImagingRig'
            MonitorID = 2;
        case '2AFCRig1'
            MonitorID = 2;
        case '2AFCRig2'
            MonitorID = 2;         
    end

    EnableMovingSpouts = 1;    
    EnableOpto         = 0;
    
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
    % [S] = m_InitGUI.UpdateOpto(S, EnableOpto);
    
    
    %% Connect Maestro
    M = [];
    if (EnableMovingSpouts == 1)
        % M = PololuMaestro('COM15');
        switch BpodSystem.Data.RigName
            case 'ImagingRig'
                M = PololuMaestro('COM5');
            case '2AFCRig1' 
                M = PololuMaestro('COM16');
            case '2AFCRig2'
                M = PololuMaestro('COM5');
                % M = PololuMaestro('COM10');
        end
        M.setMotor(0, m_TrialConfig.ConvertMaestroPos(S.GUI.RightServoInPos - S.GUI.ServoDeflection));
        M.setMotor(1, m_TrialConfig.ConvertMaestroPos(S.GUI.LeftServoInPos + S.GUI.ServoDeflection));
    end
    
    
    %% Trial initilization
    
    % set max number of trials
    
    BpodSystem.Data.TrialTypes = [];
    BpodSystem.Data.ProbeTrials = [];
    BpodSystem.Data.IsAntiBiasProbeTrial = [];
    BpodSystem.Data.MoveCorrectSpout= [];    
    BpodSystem.Data.OptoType = [];
    BpodSystem.Data.ProcessedSessionData = {};
    
    % initialize anti-bias variables
    AntiBiasVar.IncorrectFlag       = 0;
    AntiBiasVar.IncorrectType       = 1;
    AntiBiasVar.CompletedHist.left  = [];
    AntiBiasVar.CompletedHist.right = [];
    AntiBiasVar.BiasIndex           = 0;
    AntiBiasVar.ValveFlag           = 'NoBias';
    % AntiBiasVar.ServoAdjust         = 0;
    AntiBiasVar.ServoRightAdjust    = 0;
    AntiBiasVar.ServoRightTrialsSinceAdjust     = 20;
    AntiBiasVar.ServoLeftAdjust     = 0;
    AntiBiasVar.ServoLeftTrialsSinceAdjust     = 20;
    AntiBiasVar.IsAntiBiasProbeTrial = false;
    AntiBiasVar.AutoMoveSpout = false;
    AntiBiasVar.MoveCorrectSpout     = false;
    AntiBiasVar.NumSpoutSelectTrials = 3;
    AntiBiasVar.NumProbeTrials = 10;
    
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
    [ProbeTrials] = m_TrialConfig.GenProbeTrials(S);
    [OptoType]    = m_Opto.GenOptoType(S);

    % adjust warmup trials to have no more than 'max' number of consecutive
    % same-side trials
    % TrialTypes = m_TrialConfig.AdjustMaxConsecutiveSameSideTrials(TrialTypes);

    % Side Outcome Plot
    BpodSystem.ProtocolFigures.OutcomePlotFig = figure('Position', [50 540 1000 220],'name','Outcome plot','numbertitle','off', 'MenuBar', 'none', 'Resize', 'off');
    BpodSystem.GUIHandles.OutcomePlot = axes('Position', [.075 .35 .89 .55]);
    OutcomePlot = BpodSystem.GUIHandles.OutcomePlot;
    % TrialTypeOutcomePlot(BpodSystem.GUIHandles.OutcomePlot, 'init', TrialTypes);
    TrialTypeOutcomePlot(BpodSystem.GUIHandles.OutcomePlot, 'init', TrialTypes, OptoType, ProbeTrials);
    BpodParameterGUI('init', S); % Initialize parameter GUI plugin
    % m_Plotter.UpdateOutcomePlot(BpodSystem, TrialTypes, 0);
    m_Plotter.UpdateOutcomePlot(BpodSystem, TrialTypes, OptoType, ProbeTrials, 0);

    % update gui positions
    set(BpodSystem.ProtocolFigures.OutcomePlotFig, 'Position', [917 829 1000 220]);    
    set(BpodSystem.ProtocolFigures.ParameterGUI, 'Position', [5 49 1862 992]);    
    
    
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
    BpodSystem.PluginObjects.V.TimerMode = 0;
    BpodSystem.PluginObjects.V.play(0);
    BpodSystem.SoftCodeHandlerFunction = 'SoftCodeHandler';    
    BpodSystem.PluginObjects.V.play(6); 
    input('Set parameters and press enter to continue >', 's'); 
    S = BpodParameterGUI('sync', S);
    
    % update trial types
    [ProbeTrials] = m_TrialConfig.GenProbeTrials(S);
    [OptoType]    = m_Opto.GenOptoType(S);    
    m_Plotter.UpdateOutcomePlot(BpodSystem, TrialTypes, OptoType, ProbeTrials, 0);

    debug = false;
    if debug
        S.GUI.SetManualITI = 1;
        S.GUI.ManualITI = 0;
        S.GUI.PostRewardDelay = 0;
        S.GUI.ManuallTimeOutPunish = 1;
        S.GUI.TimeOutPunish = 0;
        S.GUI.NumNaiveWarmup = 0;
    end
    
    %% Main trial loop
    
    for currentTrial = 1:S.GUI.MaxTrials
        BpodSystem.Data.CurrentTrial = currentTrial;    % update these later to only use the bpod system struct
    
        S = BpodParameterGUI('sync', S);
    
        %% anti bias   
    
        % if repeatedincorrect anti-bias enabled, and previous trial was
        % punish, then draw probabilities for current trial based on trial
        % type (left or right)
        [TrialTypes, AntiBiasVar] = m_TrialConfig.RepeatedIncorrect( ...
            BpodSystem, S, AntiBiasVar, currentTrial, TrialTypes);


        [AntiBiasVar, LeftValveAmount_uL, RightValveAmount_uL, TrialTypes] = m_TrialConfig.AntiBiasValveAdjust( ...
            BpodSystem, S, AntiBiasVar, currentTrial, TrialTypes);
    
        % LeftValveAmount_uL = S.GUI.LeftValveAmount_uL;
        % RightValveAmount_uL = S.GUI.RightValveAmount_uL;


        % [TrialTypes] = m_TrialConfig.ManualFraction( ...
        %     S, currentTrial, TrialTypes); 

        % if S.GUI.ManualSideAct
        %     switch S.GUI.ManualSide
        %         case 1
        %             TrialTypes(currentTrial) = 1;
        %         case 2
        %             TrialTypes(currentTrial) = 2;
        %     end
        % end        

        % if naive and mouse doesn't lick water from spout, set trial type
        % to same as previous
        if ((currentTrial > 1) && ...
                (S.GUI.TrainingLevel == 1) && ...
                (isfield(BpodSystem.Data.RawEvents.Trial{currentTrial-1}.States, 'PunishNaive') && ...
                ~isnan(BpodSystem.Data.RawEvents.Trial{currentTrial-1}.States.PunishNaive(1))))
            TrialTypes(currentTrial) = TrialTypes(currentTrial-1);
        end

        BpodSystem.Data.TrialTypes(currentTrial) = TrialTypes(currentTrial);

        
        [AntiBiasVar, LeftValveAmount_uL, RightValveAmount_uL] = m_TrialConfig.AntiBiasProbeTrials( ...
            BpodSystem, S, AntiBiasVar, currentTrial, TrialTypes, LeftValveAmount_uL, RightValveAmount_uL);
       
        [AntiBiasVar] = m_TrialConfig.ManualSingleSpout( ...
            BpodSystem, S, AntiBiasVar);
        
        BpodSystem.Data.IsAntiBiasProbeTrial(currentTrial) = AntiBiasVar.IsAntiBiasProbeTrial;
        BpodSystem.Data.MoveCorrectSpout(currentTrial) = AntiBiasVar.MoveCorrectSpout;
        
        LeftValveTime   = m_TrialConfig.Amount2Time(LeftValveAmount_uL, 1);
        RightValveTime  = m_TrialConfig.Amount2Time(RightValveAmount_uL, 3);
    


        % 
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
        DURA.PostVisStimDelay = m_TrialConfig.GetPostVisStimDelay(S);
        
    
        %% set vis stim perturbation ISI duration according to trial-specific difficulty level
    
        % perturbation sampling
        [PrePertISI, PostPertISI, EasyMaxInfo] = m_TrialConfig.GetPostPertISI( ...
                S, TrialDifficulty, PerturbInterval, TrialTypes, currentTrial);

        % [GrayPerturbISI] = m_TrialConfig.SetPostPertISI( ...
        %     S, TrialTypes, currentTrial, PostPertISI);
        GrayPerturbISI = PostPertISI;
        GrayPrePerturbISI = PrePertISI;
    
        BpodSystem.Data.TrialVars.Trial{currentTrial}.PostPertISI = PostPertISI + S.GUI.GratingDur_s; % not sure, maybe need for data analysis to compare to prev 2afc
        BpodSystem.Data.TrialVars.Trial{currentTrial}.PostPertISIWithoutGrating = PostPertISI; 

        BpodSystem.Data.TrialVars.Trial{currentTrial}.PrePertISI = PrePertISI + S.GUI.GratingDur_s; % not sure, maybe need for data analysis to compare to prev 2afc
        BpodSystem.Data.TrialVars.Trial{currentTrial}.PrePertISIWithoutGrating = PrePertISI;        

        %% construct preperturb vis stim videos and audio stim base for grating and gray if duration parameters changed
        
        % config audio stimulus
        m_AVstimConfig.ConfigHifi(H, S, SF, Envelope);
    
        [JitterFlag] = m_TrialConfig.GetJitterFlag(S);
    
        % generate video
        VisStim.ProcessedData.Seq = [];
        VisStim.ProcessedData.PrePost = [];
        [VisStim] = m_AVstimConfig.GetVisStimImg(S, BpodSystem, FPS, VisStim);

        VisStim = m_AVstimConfig.GetVideoDataPre(S, BpodSystem, FPS, VisStim, GrayPrePerturbISI, JitterFlag);
        VisStim = m_AVstimConfig.GetVideoDataPost(S, BpodSystem, FPS, VisStim, GrayPerturbISI, JitterFlag);
        % VisStim = m_AVstimConfig.GetVideoDataExtra(S, BpodSystem, FPS, VisStim, GrayPerturbISI, JitterFlag);
    
        % combine full video
        VisStim.Data.Full = m_AVstimConfig.GetFullVideo(S, VisStim, FPS);
    
        % add gray frames for opto
        % OptoGrayFrames = [m_AVstimConfig.GetUnitVideo(VisStim.Img.GrayFrame_SyncW, 1) m_AVstimConfig.GetUnitVideo(VisStim.Img.GrayFrame_SyncBlk, 1)];
        % VisStim.Data.OptoGrayDur = m_AVstimConfig.GetVideoDur(FPS, OptoGrayFrames);
        % VisStim.Data.Full = [OptoGrayFrames VisStim.Data.Full];
       
        % load constructed video into the video object
        BpodSystem.PluginObjects.V.Videos{25} = struct;
        BpodSystem.PluginObjects.V.Videos{25}.nFrames = length(VisStim.Data.Full); 
        BpodSystem.PluginObjects.V.Videos{25}.Data = VisStim.Data.Full;

        % use vis stim duration for opto timing
        % if (S.GUI.ReactionTask == 1)
        %     VisStim.Data.VisStimDuration = VisStim.Data.Pre.Dur;
        % else
        %     VisStim.Data.VisStimDuration = VisStim.Data.Pre.Dur + VisStim.Data.Post.Dur;
        % end
        % VisStim.Data.VisStimDuration = VisStim.Data.Post.Dur;
        VisStim.Data.VisStimDuration = VisStim.Data.Pre.Dur + VisStim.Data.Post.Dur;


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
        SCOA.AudStim     = m_Opto.OptoUpdateAudStimTrig(OptoType, currentTrial);

        if (EnableMovingSpouts == 1)
            SCOA.SpoutIn = {'SoftCode', 9};
        else
            SCOA.SpoutIn = {};
        end

    
        %% construct state matrix

        sma = NewStateMatrix();

        % OptoDuration = VisStim.Data.Pre.Dur + VisStim.Data.Post.Dur + VisStim.Data.Extra.Dur;
        OptoDuration = VisStim.Data.Post.Dur;
        sma = m_Opto.SetOpto(BpodSystem, S, sma, OptoDuration, OptoType, currentTrial);

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
            case 2 % early Trained
                if (currentTrial <= S.GUI.NumNaiveWarmup)
                    ExperimenterTrialInfo.TrainingLevel = 'Naive warmup';
                    StateNaive(sma, S, SCOA, TrialTarget, VisStim.Data.VisStimDuration, DURA);
                else
                    ExperimenterTrialInfo.TrainingLevel = 'Early';
                    StateEarlyTrain(sma, S, SCOA, TrialTarget, VisStim.Data.VisStimDuration, DURA);
                end                
            case {3, 4}   %3 || 4 % Mid Trained
                switch S.GUI.TrainingLevel
                    case 3
                        ExperimenterTrialInfo.TrainingLevel = 'Mid 1';
                    case 4
                        ExperimenterTrialInfo.TrainingLevel = 'Mid 2';
                end

                if (currentTrial <= S.GUI.NumNaiveWarmup)
                    ExperimenterTrialInfo.TrainingLevel = 'Naive warmup';
                    StateNaive(sma, S, SCOA, TrialTarget, VisStim.Data.VisStimDuration, DURA);
                else
                    % ExperimenterTrialInfo.TrainingLevel = 'Mid Trained';
                    StateMidTrain(sma, S, SCOA, TrialTarget, VisStim.Data.VisStimDuration, DURA);
                end
            case 5 % well trained
                if (currentTrial <= S.GUI.NumNaiveWarmup)
                    ExperimenterTrialInfo.TrainingLevel = 'Naive warmup';
                    StateNaive(sma, S, SCOA, TrialTarget, VisStim.Data.VisStimDuration, DURA);
                else
                    ExperimenterTrialInfo.TrainingLevel = 'Well';
                    StateWellTrained(sma, S, SCOA, TrialTarget, VisStim.Data.VisStimDuration, DURA);
                end
        end        
    
    
        %% add console print for experimenter trial information
    
        ExperimenterTrialInfo.TrialNumber = currentTrial;
        ExperimenterTrialInfo.GratingOrientation = VisStim.orientation(VisStim.SampleGratingIdx-1);
        ExperimenterTrialInfo.CorrectChoice = TrialTarget.CorrectChoice;
        ExperimenterTrialInfo.ReactionTask = S.GUI.ReactionTask;
        ExperimenterTrialInfo.Bias = AntiBiasVar.ValveFlag;
        ExperimenterTrialInfo.LeftValveAmount  = LeftValveAmount_uL;
        ExperimenterTrialInfo.RightValveAmount = RightValveAmount_uL;
        ExperimenterTrialInfo.PostPertISI = GrayPerturbISI;
        % ExperimenterTrialInfo.VisStimDuration = VisStim.Data.Pre.Dur + VisStim.Data.Post.Dur + VisStim.Data.Extra.Dur;
        ExperimenterTrialInfo.VisStimDuration = VisStim.Data.Post.Dur;
        % ExperimenterTrialInfo.CategoryBoundary = S.GUI.ISIOrig_s;
        ExperimenterTrialInfo.CategoryBoundary = S.GUI.ISIShortMax_s;
        ExperimenterTrialInfo.EasyMax = EasyMaxInfo;
        ExperimenterTrialInfo.ITI = DURA.ITI;
        ExperimenterTrialInfo.TimeOutPunish = DURA.TimeOutPunish;
        ExperimenterTrialInfo.ChoiceWindow = DURA.ChoiceWindow;
        % ExperimenterTrialInfo.ChangeMindDur = DURA.ChangeMindDur;
        % ExperimenterTrialInfo.PreISIinfo = VisStim.PreISIinfo;
        ExperimenterTrialInfo.PostISIinfo = VisStim.PostISIinfo;
        ExperimenterTrialInfo.Jitter = JitterFlag;
        ExperimenterTrialInfo.Omission = VisStim.OmiFlag;
    
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
            % m_Plotter.UpdateOutcomePlot(BpodSystem, TrialTypes, 1);
            m_Plotter.UpdateOutcomePlot(BpodSystem, TrialTypes, OptoType, ProbeTrials, 1);
            SavePlot;
            % if (currentTrial > 0) && (mod(currentTrial, 60) == 0)
            %     saveas(BpodSystem.GUIHandles.OutcomePlot, ['outcome_images\outcome_plot_' num2str(outcomePlotCntr) '.png']);
            %     outcomePlotCntr = outcomePlotCntr + 1;
            %     % store plot handle
            % 
            %     % tic
            %     % exportgraphics(BpodSystem.GUIHandles.OutcomePlot, 'plot2.pdf', 'ContentType', 'vector', 'Append', true);
            %     % toc
            
            % end
            StateTiming();
            SaveBpodSessionData;


        end
        HandlePauseCondition;
        if BpodSystem.Status.BeingUsed == 0

            %%Show the following information to the Experimenter
            
            % Example protocol version
            % Print the information to the console
            fprintf('\n\n');
            fprintf('%s\n', datestr(now, 'mm/dd/yy'));
            fprintf('%s\n\n', datestr(now, 'dddd'));
            fprintf('%s\n', datetime('now', 'Format', 'h:mm a'));
            fprintf('%s\n\n', S.GUI.ExperimenterInitials);
            fprintf('Protocol Version: %s\n', protocol_version);
            fprintf('%s\n', ExperimenterTrialInfo.TrainingLevel);
            fprintf('Total Trials: %s\n\n\n', num2str(currentTrial));


            % also save outcome plot when stopping trial
            % saveas(OutcomePlot, ['outcome_images\outcome_plot_' num2str(outcomePlotCntr) '.png']);
            M.setMotor(0, m_TrialConfig.ConvertMaestroPos(S.GUI.RightServoInPos));
            M.setMotor(1, m_TrialConfig.ConvertMaestroPos(S.GUI.LeftServoInPos));             
            M = [];
            BpodSystem.PluginObjects.V = [];
            BpodSystem.setStatusLED(1);
            AntiBiasVar = [];
            return
        end
    
    end
    
    M = [];
    BpodSystem.PluginObjects.V = [];
    BpodSystem.setStatusLED(1);
    AntiBiasVar = [];

catch MatlabException
    disp(MatlabException.identifier);
    disp(getReport(MatlabException));

    % err report log file
    % recording error and stack information to file
    t = datetime;
    session_date = 10000*(year(t)-2000) + 100*month(t) + day(t);
    
    % get session file name
    [SessionFilepath, SessionFileName, Ext] = fileparts(BpodSystem.Path.CurrentDataFile);

    CrashFileDir = 'C:\data analysis\behavior\error logs\';
    CrashFileName = [CrashFileDir, num2str(session_date), '_BPod-matlab_crash_log_', SessionFileName];    

    % make crash log folder if it doesn't already exist
    [status, msg, msgID] = mkdir(CrashFileDir);

    % save workspace variables associated with session
    Data = BpodSystem.Data;
    save(CrashFileName, 'Data');
    % add more workspace vars if needed

    %open file
    fid = fopen([CrashFileName, '.txt'],'a+');

    % write session associated with error
    fprintf(fid,'%s\n', SessionFileName);

    % date
    fprintf(fid,'%s\n', num2str(session_date));

    % rig specs
    fprintf(fid,'%s\n', BpodSystem.Data.RigName);

    % write the error to file   
    fprintf(fid,'%s\n',MatlabException.identifier);
    fprintf(fid,'%s\n',MatlabException.message);
    fprintf(fid,'%s\n',MatlabException.Correction);

    % print stack
    fprintf(fid, '%s', MatlabException.getReport('extended', 'hyperlinks','off'));

    % close file
    fclose(fid);

    % save workspace variables associated with session to file

    BpodSystem.PluginObjects.V = [];
    disp('Resetting maestro object...');
    BpodSystem.setStatusLED(1); % enable Bpod status LEDs after session

    % set servos to out position
    M.setMotor(0, m_TrialConfig.ConvertMaestroPos(S.GUI.RightServoInPos));
    M.setMotor(1, m_TrialConfig.ConvertMaestroPos(S.GUI.LeftServoInPos)); 
    M = [];

    AntiBiasVar = [];
end
end

% match rig ID to computer name for rig-specific settings
% (features/timing/servos/etc)
function SetRigID(BpodSystem)
    BpodSystem.Data.ComputerHostName = getenv('COMPUTERNAME');
    BpodSystem.Data.RigName = '';
    switch BpodSystem.Data.ComputerHostName
        case 'COS-3A11406'
            BpodSystem.Data.RigName = 'ImagingRig';
        case 'COS-3A11427'
            BpodSystem.Data.RigName = 'JoystickRig1';
        case 'COS-3A17904'
            BpodSystem.Data.RigName = 'JoystickRig2';
        case 'COS-3A14773'
            BpodSystem.Data.RigName = 'JoystickRig3';     
        case 'COS-3A11264'
            BpodSystem.Data.RigName = '2AFCRig2';   
        case 'COS-3A11215'
            BpodSystem.Data.RigName = '2AFCRig1';             
    end
end

function SetMotorPos = ConvertMaestroPos(obj, MaestroPosition)
    m = 0.002;
    b = -3;
    SetMotorPos = MaestroPosition * m + b;
end