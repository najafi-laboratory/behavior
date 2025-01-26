classdef InitGUI
    methods


function [S] = SetParams(obj, BpodSystem)
    S = BpodSystem.ProtocolSettings; % Load settings chosen in launch manager into current workspace as a struct called S

    if isempty(fieldnames(S))  % If settings file was an empty struct, populate struct with default settings
        S.GUI.currentTrial = 0;
        
        % joystick params
        S.GUI.ExperimenterInitials = 'Initials';
        S.GUI.SelfTimedMode = 0;
        S.GUIMeta.SelfTimedMode.Style = 'checkbox';  
        S.GUI.ChemogeneticSession = 0;
        S.GUIMeta.ChemogeneticSession.Style = 'checkbox';
        % S.GUI.mgCNO = '1';
        % S.GUI.mlSaline = '1';

        S.GUI.mgCNO_mlSaline = '1:1';   % number of required repeated lever presses in a trial

        S.GUI.Threshold = 2; % Threshold for completion of a lever press, units = degrees of shaft rotation
        S.GUI.EarlyPressThreshold = 1;
        S.GUI.WarmupThreshold = S.GUI.Threshold / 2; % Threshold for completion of a lever press, units = degrees of shaft rotation        
        S.GUI.RetractThreshold = 0.3;   
        % window setting parameters 
        S.GUI.Press1Window_s = 1.8; % how long mouse has to press lever
        S.GUI.Press2WindowShort_s = 1.8; % how long mouse has to press lever 
        S.GUI.Press2WindowLong_s = 1.8; % how long mouse has to press lever
        S.GUI.Press2WindowWarmup_s = 1.8;
        S.GUI.EnableAutoPressWinReduce = 1;
        S.GUIMeta.EnableAutoPressWinReduce.Style = 'checkbox'; 
        S.GUI.AutoPressWin1ReduceMin = 0.500;
        S.GUI.AutoPressWinShortReduceMin = 0.500;
        S.GUI.AutoPressWinLongReduceMin = 0.500;
        S.GUI.AutoPressWinReduceStep = 0.001;
    
        % set servo positions per rig, allows consistent code version
        % updates compatible across rigs
        switch BpodSystem.Data.RigName
            case 'ImagingRig'
                S.GUI.ServoInPos = 1573.00; % lever start pos
                % S.GUI.ServoInPos = 1810.00; % lever start pos
                S.GUI.ServoOutPos = 34; % can press lever
            case 'JoystickRig1'
                S.GUI.ServoInPos = 1602.00; % lever start pos
                S.GUI.ServoOutPos = 38; % can press lever
            case 'JoystickRig2'
                S.GUI.ServoInPos = 1308.00; % lever start pos
                S.GUI.ServoOutPos = 43; % can press lever   
            case 'JoystickRig3'
                S.GUI.ServoInPos = 1615.00; % lever start pos rig 3
                % S.GUI.ServoInPos = 1645.00; % lever start pos
                S.GUI.ServoOutPos = 43; % can press lever     rig 3               
                % S.GUI.ServoOutPos = 34; % can press lever     
            case 'JoystickRig4'
                S.GUI.ServoInPos = 1647.00; % lever start pos            
                S.GUI.ServoOutPos = 34; % can press lever                 
        end

        S.GUI.PrePress2DelayShort_s = 0.100;
        S.GUI.PrePress2DelayLong_s = 0.200;

        S.GUI.EnableAutoDelay = 1;
        S.GUIMeta.EnableAutoDelay.Style = 'checkbox';                
        S.GUI.AutoDelayMaxShort_s = 0.500;
        S.GUI.AutoDelayMaxLong_s = 1.000;
        S.GUI.AutoDelayStep_s = 0.0003;

        S.GUI.EnablePreVis2DelayJitter = 0;
        S.GUIMeta.EnablePreVis2DelayJitter.Style = 'checkbox';
        S.GUI.PreVis2DelayJitterStd = 0.080;  % std deviation of normal dist for drawing jitter
        S.GUI.PreVis2DelayMargin_s = 0.050;    % S.GUI.PreVis2DelayMargin_s = 0.050;
        % range cuttoff for jitter

        S.GUI.EnableManualTrialType = 0;
        S.GUIMeta.EnableManualTrialType.Style = 'checkbox';
        S.GUI.ManualTrialType = 1;
        S.GUIMeta.ManualTrialType.Style = 'popupmenu';
        S.GUIMeta.ManualTrialType.String = {'Short', 'Long'};
        % block setting parameters
        S.GUI.EnableBlockChange = 1;
        S.GUIMeta.EnableBlockChange.Style = 'checkbox';
        S.GUI.TrialTypeSequence = 2;
        S.GUIMeta.TrialTypeSequence.Style = 'popupmenu';
        S.GUIMeta.TrialTypeSequence.String = {'Random', 'Random First Block', 'Short First Block', 'Long First Block'};
        S.GUI.NumTrialsPerBlock = 20;
        S.GUI.MaxNumTrialsPerBlock = 75;
        S.GUI.BlockLengthMargin = 5;
        S.GUI.ProbeTrialFraction = 0.1;

        S.GUIPanels.Joystick = {'ExperimenterInitials', 'SelfTimedMode', 'ChemogeneticSession', 'mgCNO_mlSaline', 'Threshold', 'EarlyPressThreshold', 'WarmupThreshold', 'RetractThreshold', 'ServoInPos', 'ServoOutPos', 'PrePress2DelayShort_s', 'PrePress2DelayLong_s', 'EnableAutoDelay', 'AutoDelayMaxShort_s', 'AutoDelayMaxLong_s', 'AutoDelayStep_s', 'EnablePreVis2DelayJitter', 'PreVis2DelayJitterStd', 'PreVis2DelayMargin_s', 'EnableManualTrialType', 'ManualTrialType'};
        S.GUIPanels.WindowSetting = {'Press1Window_s', 'Press2WindowWarmup_s','Press2WindowShort_s','Press2WindowLong_s', 'EnableAutoPressWinReduce','AutoPressWin1ReduceMin' ,'AutoPressWinShortReduceMin','AutoPressWinLongReduceMin', 'AutoPressWinReduceStep'} ;
        %S.GUIPanels.BlockSetting = {'TrialTypeSequence', 'NumTrialsPerBlock','MaxNumTrialsPerBlock', 'BlockLengthMargin', 'ProbeTrialFraction'};
        % S.GUIPanels.Joystick = {'SelfTimedMode', 'ChemogeneticSession', 'mgCNO', 'mlSaline', 'mgCNO_mlSaline', 'Threshold', 'WarmupThreshold', 'RetractThreshold', 'EarlyPressThreshold', 'Press1Window_s', 'Press2Window_s', 'PressWindowExtend_s',  'EnableAutoPressWinReduce', 'AutoPressWinReduceMin', 'AutoPressWinReduceStep', 'Reps', 'ServoInPos', 'ServoOutPos', 'PrePress2DelayShort_s', 'PrePress2DelayLong_s', 'EnableAutoDelay', 'AutoDelayMaxShort_s', 'AutoDelayMaxLong_s', 'AutoDelayStep_s', 'EnablePreVis2DelayJitter', 'PreVis2DelayJitterStd', 'PreVis2DelayMargin_s', 'EnableManualTrialType', 'ManualTrialType', 'TrialTypeSequence', 'NumTrialsPerBlock', 'BlockLengthMargin', 'ProbeTrialFraction'};
        % S.GUIPanels.Joystick = {'ChemogeneticSession', 'mgCNO', 'mlSaline', 'Threshold', 'WarmupThreshold', 'EnableAutoPressWinReduce', 'AutoPressWinReduceStep', 'AutoPressWinReduceMin', 'Press1Window_s', 'Press2Window_s', 'PressWindowExtend_s', 'Reps', 'ZeroRTrials', 'ResistanceLevel', 'ServoInPos', 'ServoOutPos', 'RetractThreshold', 'EarlyPressThreshold', 'SelfTimedMode', 'PrePress2DelayShort_s', 'PrePress2DelayLong_s', 'PreVis2DelayMargin_s', 'EnableAutoDelay', 'AutoDelayStep_s', 'AutoDelayMaxShort_s', 'AutoDelayMaxLong_s', 'EnableManualTrialType', 'ManualTrialType', 'TrialTypeSequence', 'NumTrialsPerBlock', 'BlockLengthMargin', 'ProbeTrialFraction'};

        % Optogentic params
        S.GUI.SessionType = 2;  % S.GUI.SessionType = 2;
        S.GUIMeta.SessionType.Style = 'popupmenu';
        S.GUIMeta.SessionType.String = {'Opto', 'Control'};
        S.GUI.PulseType = 3;
        S.GUIMeta.PulseType.Style = 'popupmenu';
        S.GUIMeta.PulseType.String = {'ContinuousOpto', 'SquareWaveOpto_ContinuousShutter', 'SquareWaveOpto_PulsedShutter'};
        S.GUI.MaxOptoDur_s = 2.5;
        S.GUI.LEDOnPulseDur_ms = 7.8;
        S.GUI.LEDOffPulseDur_ms = 92.2;
        S.GUI.OptoVis1 = 1;
        S.GUIMeta.OptoVis1.Style = 'checkbox';

        S.GUI.OptoVis2 = 1;
        S.GUIMeta.OptoVis2.Style = 'checkbox';

        S.GUI.OptoPrePressDelay = 1;
        S.GUIMeta.OptoPrePressDelay.Style = 'checkbox';

        S.GUI.OptoWaitForPress1 = 1;
        S.GUIMeta.OptoWaitForPress1.Style = 'checkbox';


        S.GUI.OptoWaitForPress2 = 1;
        S.GUIMeta.OptoWaitForPress2.Style = 'checkbox';        

        S.GUI.OptoPress1 = 1;
        S.GUIMeta.OptoPress1.Style = 'checkbox';

        S.GUI.OptoPress2 = 1;
        S.GUIMeta.OptoPress2.Style = 'checkbox';

        S.GUI.OptoITI = 1;
        S.GUIMeta.OptoITI.Style = 'checkbox';

        S.GUI.OptoITICycleType = 2; % S.GUI.OptoTrialTypeSeq = 5;
        S.GUIMeta.OptoITICycleType.Style = 'popupmenu';
        S.GUIMeta.OptoITICycleType.String = {'NumPulses', 'Duration'};

        S.GUI.OptoITINumPulses = 5;
        S.GUI.OptoITIDur_s = 1;

        S.GUI.OptoITIAlignment = 1; % S.GUI.OptoTrialTypeSeq = 5;
        S.GUIMeta.OptoITIAlignment.Style = 'popupmenu';
        S.GUIMeta.OptoITIAlignment.String = {'StateStart', 'StateStop'};                       

        S.GUI.OptoTrialTypeSeq = 5; % S.GUI.OptoTrialTypeSeq = 5;
        S.GUIMeta.OptoTrialTypeSeq.Style = 'popupmenu';
        S.GUIMeta.OptoTrialTypeSeq.String = {'Random', 'Random First Block', 'Off First Block', 'On First Block', 'On Epoch'};               
        S.GUI.OnFraction = 0.35;    % S.GUI.OnFraction = 0.35;         
        S.GUI.EpochTrialStart = 1;
        S.GUI.EpochTrialStop = 15;
        S.GUI.NumOptoTrialsPerBlock = 50;
        S.GUIPanels.Opto = {'SessionType', 'PulseType', 'MaxOptoDur_s', 'LEDOnPulseDur_ms', 'LEDOffPulseDur_ms','OptoVis1', 'OptoVis2', 'OptoPrePressDelay', 'OptoWaitForPress1', 'OptoWaitForPress2', 'OptoPress1', 'OptoPress2', 'OptoITI', 'OptoITICycleType', 'OptoITINumPulses', 'OptoITIDur_s', 'OptoITIAlignment', 'OptoTrialTypeSeq', 'OnFraction', 'EpochTrialStart', 'EpochTrialStop', 'NumOptoTrialsPerBlock'};
        S.GUIPanels.BlockSetting = {'EnableBlockChange','TrialTypeSequence', 'NumTrialsPerBlock','MaxNumTrialsPerBlock', 'BlockLengthMargin', 'ProbeTrialFraction'};



        % reward
        S.GUI.PreRewardDelay_s = 0.100;
        S.GUI.EnableAutoPreRewardDelay = 1;
        S.GUIMeta.EnableAutoPreRewardDelay.Style = 'checkbox';                        
        S.GUI.AutoPreRewardDelayMax_s = 0.200;
        S.GUI.AutoPreRewardDelayStep_s = 0.0001;
        S.GUI.PostRewardDelay_s = 1.000; % post reward delay prior to lever return
        S.GUI.CenterValveAmount_uL = 1;
        S.GUIPanels.Reward = {'PreRewardDelay_s', 'EnableAutoPreRewardDelay', 'AutoPreRewardDelayMax_s', 'AutoPreRewardDelayStep_s', 'PostRewardDelay_s', 'CenterValveAmount_uL'};        
        

        % ITI params
        S.GUI.ITI_Pre = 0.500;  % V_3_3 ITI_post, fixed duration, now ITI_Pre V_3_7
        S.GUI.SetManualITI = 0; % S.GUI.SetManualITI = 0;
        S.GUIMeta.SetManualITI.Style = 'checkbox';
        S.GUI.ManualITI = '0';
        S.GUI.ITIMin = 1;    % Minimum ITI (in seconds)
        S.GUI.ITIMax = 5;    % Maximum ITI (in seconds)
        S.GUI.ITIMean = 3;   % Mean ITI (in seconds)
        S.GUI.PunishITI = 1;  % early press and didnt press punish ITI (from EarlyPress1Punish, EarlyPress2Punish, Punish)
        S.GUIPanels.ITI_Dist = {'ITI_Pre', 'SetManualITI', 'ManualITI', 'ITIMin', 'ITIMax', 'ITIMean', 'PunishITI'};
    



        % excluded trials
        S.GUI.ExcludedTrials = @(src,event)fnExcludedTrials;
        S.GUIMeta.ExcludedTrials.Style = 'pushbutton';
        S.GUI.numExcludedTrials = 0;    
        S.GUIPanels.ExcludedTrials = {'ExcludedTrials', 'numExcludedTrials'};

        % assisted trials
        S.GUI.AssistedTrials = @(src,event)fnAssistedTrials;
        S.GUIMeta.AssistedTrials.Style = 'pushbutton';
        S.GUI.ATRangeStart = 0;
        S.GUI.ATRangeStop = 0;
        S.GUIPanels.AssistedTrials = {'AssistedTrials', 'ATRangeStart', 'ATRangeStop'};

        % init cue params
                      
        % Reward Pulse
        S.GUI.EnableRewardPulses = 0;
        S.GUIMeta.EnableRewardPulses.Style = 'checkbox';
                            
        S.GUI.NumPulses = 15;
        S.GUI.PulseAmount_uL = 1;
        S.GUI.RewardIntervalDist = 1;
        S.GUIMeta.RewardIntervalDist.Style = 'popupmenu';
        S.GUIMeta.RewardIntervalDist.String = {'Fixed', 'Exponential'};        
        S.GUI.RewardIntervalFixed = 3;
        S.GUI.RewardIntervalMin = 2;    % Minimum NumPulses (in seconds)
        S.GUI.RewardIntervalMax = 5;    % Maximum NumPulses (in seconds)
        S.GUI.RewardIntervalMean = 3;   % Mean NumPulses (in seconds)     
        % S.GUIPanels.RewardPulse = {'EnableRewardPulses', 'NumPulsesDist', 'NumPulses', 'NumPulsesMin', 'NumPulsesMax', 'NumPulsesMean', 'PulseAmountDist', 'PulseAmount_uL','PulseAmountMin','PulseAmountMax','PulseAmountMean', 'IPIDist', 'IPI_s','IPIMin','IPIMax','IPIMean'};
        S.GUIPanels.RewardPulse = {'EnableRewardPulses', 'NumPulses', 'PulseAmount_uL', 'RewardIntervalDist', 'RewardIntervalFixed','RewardIntervalMin','RewardIntervalMax','RewardIntervalMean'};


        % training level params
        S.GUI.MaxTrials = 1000;
        % S.GUI.TrainingLevel = 5;
        % S.GUIMeta.TrainingLevel.Style = 'popupmenu'; % the GUIMeta field is used by the ParameterGUI plugin to customize UI objects.
        % S.GUIMeta.TrainingLevel.String = {'Habituation', 'Naive', 'Mid Trained 1', 'Mid Trained 2', 'Well Trained'};
        S.GUI.NumEasyWarmupTrials = 10;
        S.GUIPanels.Training = {'MaxTrials', 'NumEasyWarmupTrials'};
    
        % difficulty params       
    
        % audio stim
        S.GUI.AudioStimEnable = 0;
        S.GUIMeta.AudioStimEnable.Style = 'checkbox';
        S.GUI.AudioStimVolume_percent = 1;  % volume control
        S.GUI.AudioStimFreq_Hz = 11025; % Frequency of audio stim, even multiple of SF = 44100
        S.GUIPanels.AudioStim = {'AudioStimEnable', 'AudioStimVolume_percent', 'AudioStimFreq_Hz'};
    
        % vis stim params
        S.GUI.VisStimEnable = 1;
        S.GUIMeta.VisStimEnable.Style = 'checkbox';
        S.GUI.GratingDur_s = 0.1; % Duration of grating stimulus in seconds - UPDATE
        S.GUI.ISIOrig_s = 0.5; % Duration of *fixed* gray screen stimulus in seconds - UPDATE
        S.GUI.ISIOrig_s = 1; % Duration of *fixed* gray screen stimulus in seconds - UPDATE
        S.GUI.PreVisStimDelay_s = 0; % How long the mouse must poke in the center to activate the goal port
        % S.GUI.PreGoCueDelay_s = 0;
        S.GUIPanels.VisStim = {'VisStimEnable', 'GratingDur_s', 'ISIOrig_s', 'PreVisStimDelay_s'};         

        % punish params
        S.GUI.IncorrectSoundVolume_percent = 0.07;  % volume control
        S.GUI.PunishSoundDuration_s = 1; % Seconds to wait on errors before next trial can start
        S.GUI.IncorrectSound = 1; % if 1, plays a white noise pulse on error. if 0, no sound is played.
        S.GUIMeta.IncorrectSound.Style = 'checkbox';
        S.GUI.EarlyPressPunishSoundVolume_percent = 0.00;
        S.GUI.EarlyPressPunishSoundDuration_s = 1;
        

        S.GUIPanels.Punish = {'IncorrectSoundVolume_percent', 'PunishSoundDuration_s', 'IncorrectSound', 'EarlyPressPunishSoundVolume_percent', 'EarlyPressPunishSoundDuration_s'}; 
    
    end
end
    end
end
