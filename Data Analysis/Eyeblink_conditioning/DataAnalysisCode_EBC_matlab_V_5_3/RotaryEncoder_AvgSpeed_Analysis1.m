clc; clear; close all;

%% ===================== USER SETTINGS =====================
% data_files = dir('YH26LG_EBC_V_5_0_20251209_122442.mat');
data_files = dir('*_EBC_*.mat');

% Alignment
t_pre  = 0.5;      % seconds before LED
t_post = 2.0;      % seconds after LED
dt     = 0.005;    % common time step (200 Hz); adjust if you want
tvec   = -t_pre:dt:t_post;


% Wheel signal choice (your screenshot shows these fields exist)
% We will prefer: LinearPositions (best), else Positions
WHEEL_FIELD_PREFERENCE = {'LinearPositions','Positions','PositionsUnwrapped'};

% Running/stationary classification window relative to LED
runWin = [0.0 2.0];   % seconds after LED to compute mean speed

% Running threshold (units depend on your LinearPositions units)
% You should tune this after looking at trial-by-trial speed.
runSpeedThresh = 1.0; % e.g., 1 "unit/s"

% If you already have CR labels in your code, set this to 'external'
% and provide `CR_isGood` per trial below.
CR_SOURCE = 'external';  % 'external' or 'try_from_SessionData'

% ---- If external: put your per-trial CR boolean here (must match #trials)
% Example placeholder (WILL BE REPLACED if you load your own):
CR_isGood = [];  % e.g., CR_isGood = myGoodCRLogical(:);4444
    % ===== POOLED (across sessions) storage =====
    pooledShortWheel = [];   % will become (NshortTrialsTotal x length(tvec))
    pooledLongWheel  = [];   % will become (NlongTrialsTotal  x length(tvec))

%% ===================== MAIN LOOP =====================
for f = 1:numel(data_files)

    S = load(data_files(f).name);
    if ~isfield(S,'SessionData')
        warning('No SessionData in %s, skipping.', data_files(f).name);
        continue;
    end
    SessionData = S.SessionData;

    % --- Determine trial count
    nTrials = numel(SessionData.RawEvents.Trial);

    % --- Get LED onset per trial (GlobalTimer1)
    tLED = nan(nTrials,1);
    for ctr_trial = 1:nTrials
        try
            tLED(ctr_trial) = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer1_Start;
        catch
            tLED(ctr_trial) = NaN;
        end
    end

    % Initialize an empty array to store all eyeAreaPixels values
    % allEyeAreaPixels = [];    
    % % Loop through each trial to collect eyeAreaPixels data
    % for trialIdx = 1:nTrials   
    %     eyeAreaPixels = SessionData.RawEvents.Trial{1, trialIdx}.Data.eyeAreaPixels;
    %     allEyeAreaPixels = [allEyeAreaPixels, eyeAreaPixels]; % Concatenate data 
    % end
    % % Find the overall maximum value across all collected eyeAreaPixels
    % overallMax = max(allEyeAreaPixels);

    % % --- Get wheel time series per trial (aligned to LED)
    % [wheelPos, wheelSpd] = get_wheel_aligned(SessionData, tLED, tvec, WHEEL_FIELD_PREFERENCE);
    % 
    % --- Short/Long labels
    % [isShort, isLong] = infer_short_long_trials(SessionData, nTrials);

    % ============================================================
    % --- CR labels (REVISED): compute using your classify_CR_05
    % ============================================================
    CR_isGood = false(nTrials,1);          % Good CR = true
    CR_category_all = strings(nTrials,1);  % 'Good CR' / 'Poor CR' / 'No CR'

    good_CR_threshold = 0.05;
    poor_CR_threshold = 0.02;


idxLong  = [];
idxShort = [];
idxWarmup = [];


    for ctr_trial = 1:nTrials

        % Skip trials without LED
        if ~isfinite(tLED(ctr_trial))
            continue;
        end

        % --- LED timing aligned to LED onset (t=0 at LED start)
        try
            LED_Onset     = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer1_Start;
            LED_Onset_End = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer1_End;
            AirPuff_Start = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer2_Start;
            AirPuff_End = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer2_End;
        
            LED_Onset_Zero_Start = LED_Onset - LED_Onset;
            LED_Onset_Zero_End = LED_Onset_End - LED_Onset;
            AirPuff_LED_Onset_Aligned_Start = AirPuff_Start - LED_Onset;
            AirPuff_LED_Onset_Aligned_End = AirPuff_End - LED_Onset;
        catch
            continue;
        end
        fps = 250; % frames per second, frequency of images
        seconds_before = 0.5;
        seconds_after = 0.6;

        Frames_before = fps * seconds_before;
        Frames_after = fps * seconds_after;

        LED_Puff_ISI_start = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1);
        LED_Puff_ISI_end = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2);
       % Determine a common time vector for interpolation
        common_time_vector = linspace(-seconds_before, seconds_after, Frames_before + Frames_after + 1);
        FEC_led_aligned = SessionData.RawEvents.Trial{1, ctr_trial}.Data.FECTimes - LED_Puff_ISI_start;
        % FEC_norm = 1 - SessionData.RawEvents.Trial{1, ctr_trial}.Data.eyeAreaPixels /overallMax;
        FEC_norm = SessionData.RawEvents.Trial{1, ctr_trial}.Data.FEC;

       % Initialize a matrix to store interpolated FEC data
        FEC_norm_matrix = zeros(nTrials, length(common_time_vector));
    
        abs_FEC_led_aligned = abs(FEC_led_aligned);
        closest_frame_idx_to_LED_Onset = find(abs_FEC_led_aligned == min(abs_FEC_led_aligned));

        start_idx = closest_frame_idx_to_LED_Onset - Frames_before;
        stop_idx = closest_frame_idx_to_LED_Onset + Frames_after;

        len_FEC_led_aligned = length(FEC_led_aligned);
        start_idx = max(1, min(start_idx, len_FEC_led_aligned));
        stop_idx = max(1, min(stop_idx, len_FEC_led_aligned));

        FEC_led_aligned_trimmed = FEC_led_aligned(start_idx : stop_idx);
        FEC_trimmed = FEC_norm(start_idx : stop_idx);
        
        % Apply smoothing to reduce noise
        FEC_led_aligned_trimmed_smooth = smoothdata(FEC_led_aligned_trimmed, 'movmean', 5); % Moving average
        FEC_trimmed_smooth = smoothdata(FEC_trimmed, 'movmean', 5); % Moving average

        % isLong = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1) > 0.3;
        % isShort = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1) < 0.3;
        isLong  = false(nTrials,1);
        isShort = false(nTrials,1);


        try
            blk = SessionData.RawEvents.Trial{1, ctr_trial}.Data.BlockType;
    
            if strcmpi(blk, 'long')
                isLong(ctr_trial) = true;
                idxLong(end+1) = ctr_trial;
    
            elseif strcmpi(blk, 'warm_up')
                % treat warmup as long if you want
                isLong(ctr_trial) = true;
                idxLong(end+1) = ctr_trial;
                idxWarmup(end+1) = ctr_trial;
    
            elseif strcmpi(blk, 'short')
                isShort(ctr_trial) = true;
                idxShort(end+1) = ctr_trial;
            end
    
        catch
            % BlockType missing → ignore trial
        end
        
        t_LED = LED_Onset_Zero_End;
        t_puff = AirPuff_LED_Onset_Aligned_Start;
        % --- Call your classifier
        % NOTE: classify_CR_05 signature is:
        %   classify_CR_05(time, signal, t_LED, t_puff, goodThr, poorThr)
        CR_category = classify_CR_05( ...
            FEC_led_aligned_trimmed_smooth, FEC_trimmed_smooth,...
            t_LED, ...
            t_puff, ...
            good_CR_threshold, ...
            poor_CR_threshold);

        CR_category_all(ctr_trial) = CR_category;
% || strcmp(CR_category, 'Poor CR') || strcmp(CR_category, 'No CR')
        % if strcmp(CR_category, 'Good CR') 
            CR_isGood(ctr_trial) = true;
        % end
    end

    % Optional sanity print
    fprintf('\n=== %s ===\n', data_files(f).name);
    fprintf('Good CR trials: %d / %d (%.1f%%)\n', sum(CR_isGood), nTrials, 100*mean(CR_isGood));

    % % --- Analysis 1: plot aligned wheel (short vs long)
    % fig1 = figure('Color','w','Name',['Wheel aligned on LED: ' data_files(f).name], ...
    %               'Position',[200 120 1200 450]);
    % tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

    [wheelPos, wheelSpd, ...
     wheelPosShort, wheelSpdShort, ...
     wheelPosLong,  wheelSpdLong] = ...
    get_wheel_aligned_1(SessionData, tLED, tvec, WHEEL_FIELD_PREFERENCE, idxShort, idxLong);

    wheelSpdShortToPlot = abs(wheelSpdShort);
    wheelSpdLongToPlot  = abs(wheelSpdLong);
    % --- Add this session's trials into pooled arrays
    pooledShortWheel = [pooledShortWheel; wheelSpdShort]; %#ok<AGROW>
    pooledLongWheel  = [pooledLongWheel;  wheelSpdLong];  %#ok<AGROW>

    % pooledShortWheel = [pooledShortWheel; wheelSpdToPlot(idxShort,:)]; %#ok<AGROW>
    % pooledLongWheel  = [pooledLongWheel;  wheelSpdToPlot(isLong,:)];  %#ok<AGROW>

end





    % ===== POOLED PLOT ACROSS ALL SESSIONS =====
    figure('Color','w','Name','POOLED Wheel aligned to LED','Position',[200 120 1200 450]);
    tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
    
    % nexttile;
    % plot_trialstack_mean_median(tvec, pooledShortWheel, 'POOLED SHORT: Angular speed from encoder Positions aligned to LED');
    % 
    % nexttile;
    % plot_trialstack_mean_median(tvec, pooledLongWheel, 'POOLED LONG: Angular speed from encoder Positions aligned to LED');
    
    % nexttile;
    % plot_mean_sem_and_median_iqr(tvec, pooledShortWheel, 'POOLED SHORT');
    % nexttile;
    % plot_mean_sem_and_median_iqr(tvec, pooledLongWheel, 'POOLED LONG');

    LED_win = [0.00 0.05];   % 50 ms LED
    
    Puff_short = [0.20 0.22];   % short airpuff
    Puff_long  = [0.40 0.42];   % long airpuff
    
    nexttile;
    plot_mean_sem_and_median_iqr_1(tvec, pooledShortWheel, 'POOLED SHORT', ...
        'LED',  LED_win, ...
        'Puff', Puff_short, ...
        'PuffColor', [0.4 0.6 1.0]);   % BLUE
    
    nexttile;
    plot_mean_sem_and_median_iqr_1(tvec, pooledLongWheel, 'POOLED LONG', ...
        'LED',  LED_win, ...
        'Puff', Puff_long, ...
        'PuffColor', [0.4 0.8 0.4]);   % GREEN

    sgtitle('POOLED across all sessions');
    % Save the figure
    [~, name, ~] = fileparts(data_files(f).name);
    nameParts = split(name, '_');
    if length(nameParts) >= 6
        prefixPart = nameParts{1};
        % Specify the desired file format as 'pdf'
        fileFormat = '.pdf';  % PDF format
          
        newFilename = sprintf('%s_RotaryEncoder_Mean', prefixPart); 

        newFilename = [newFilename, fileFormat];
        exportgraphics(gcf, newFilename, 'ContentType', 'vector');
    else
        error('Filename does not have the expected format');
    end