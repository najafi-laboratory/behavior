clc; clear; close all;

% ---------------- Select sessions ----------------
% Option A: all matching files in folder
data_files = dir('*_EBC_*.mat');

% Option B: specify explicitly
% data_files = dir('SM09_EBC_V_5_3_20260108_121651.mat');
dateTok = regexp({data_files.name}, '\d{8}', 'match', 'once');
sessionDates = datetime(dateTok, 'InputFormat','yyyyMMdd');
sessionDates = sessionDates(~isnat(sessionDates));

firstDate = datestr(min(sessionDates), 'mm/dd/yyyy');
lastDate  = datestr(max(sessionDates), 'mm/dd/yyyy');
% ---------------- Settings ----------------
cfg.t_pre  = 0.2;
cfg.t_post = 0.6;
cfg.dt     = 1/250;        % common grid step
cfg.smoothWin = 5;         % 0 = none
cfg.excludeProbe = true;
cfg.excludeTimeout = true;

cfg.shortISI_max = 0.30;   % <= 0.30 s => short, else long
cfg.interpMethod = 'linear';

% Pooling mode:
%   'trials'   -> pool all trials across sessions (trial-weighted)
%   'sessions' -> compute mean per session, then average sessions (session-weighted)
cfg.poolMode = 'trials';

% ---------------- Run pooling ----------------
pooled = pooledAvgFEC_fromFiles(data_files, cfg);

% ---------------- Plot ----------------
figure('Position',[100 100 1100 450]);
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

% SHORT
nexttile; hold on;
plot(pooled.short.t, pooled.short.mean, 'LineWidth', 1.8);
fill([pooled.short.t fliplr(pooled.short.t)], ...
     [pooled.short.mean + pooled.short.sem, fliplr(pooled.short.mean - pooled.short.sem)], ...
     'k', 'FaceAlpha', 0.15, 'EdgeColor', 'none');

xline(0,'--','LED','LabelVerticalAlignment','bottom','LabelHorizontalAlignment','left');
if ~isnan(pooled.short.airpuff_t)
    xline(pooled.short.airpuff_t,'--','Airpuff','LabelVerticalAlignment','bottom','LabelHorizontalAlignment','left');
end
title(sprintf('Pooled SHORT (%s) | sessions=%d | trials=%d', ...
      cfg.poolMode, pooled.nSessionsUsed, pooled.short.nTrials), ...
      'Interpreter','latex', ...
      'FontSize',14);
xlabel('Time from LED onset (s)'); ylabel('FEC');
xlim([-cfg.t_pre cfg.t_post]); grid off;

% LONG
nexttile; hold on;
plot(pooled.long.t, pooled.long.mean, 'LineWidth', 1.8);
fill([pooled.long.t fliplr(pooled.long.t)], ...
     [pooled.long.mean + pooled.long.sem, fliplr(pooled.long.mean - pooled.long.sem)], ...
     'k', 'FaceAlpha', 0.15, 'EdgeColor', 'none');

xline(0,'--','LED','LabelVerticalAlignment','bottom','LabelHorizontalAlignment','left');
if ~isnan(pooled.long.airpuff_t)
    xline(pooled.long.airpuff_t,'--','Airpuff','LabelVerticalAlignment','bottom','LabelHorizontalAlignment','left');
end
title(sprintf('Pooled LONG (%s) | sessions=%d | trials=%d', ...
      cfg.poolMode, pooled.nSessionsUsed, pooled.long.nTrials), ...
      'Interpreter','latex', ...
      'FontSize',14);

legend(sprintf('Sessions:\n%s -- %s', firstDate, lastDate), ...
       'Location','bestoutside', ...
       'Interpreter','latex', ...
       'FontSize',12, ...
       'Box','off');

xlabel('Time from LED onset (s)'); ylabel('FEC');
xlim([-cfg.t_pre cfg.t_post]); grid off;
set(findall(gcf,'Type','axes'), 'TickDir', 'out');
% ---------------- Save as PDF ----------------

firstTok = char(min(sessionDates), 'yyyyMMdd');
lastTok  = char(max(sessionDates), 'yyyyMMdd');

exportgraphics(gcf, sprintf('PooledAvgFEC_%s_%s_to_%s.pdf', cfg.poolMode, firstTok, lastTok), ...
    'ContentType','vector');
%% =================== Functions (local) ===================

function pooled = pooledAvgFEC_fromFiles(data_files, cfg)

    % Common grid
    tGrid = -cfg.t_pre:cfg.dt:cfg.t_post;

    % Accumulators for trial-pooled
    allShortTrials = [];
    allLongTrials  = [];
    allShortPuff   = [];
    allLongPuff    = [];

    % Accumulators for session-pooled (store per-session means)
    shortSessionMeans = [];
    longSessionMeans  = [];
    shortSessionNs    = [];
    longSessionNs     = [];
    shortSessionPuff  = [];
    longSessionPuff   = [];

    nSessionsUsed = 0;

    for i = 1:numel(data_files)
        S = load(data_files(i).name);
        if ~isfield(S,'SessionData')
            continue;
        end
        SessionData = S.SessionData;

        % Compute per-session short/long trial matrices (LED-aligned)
        [sessShortTrials, sessLongTrials, sessShortPuff, sessLongPuff] = ...
            extractShortLongTrials(SessionData, tGrid, cfg);

        % If session has nothing, skip
        if isempty(sessShortTrials) && isempty(sessLongTrials)
            continue;
        end
        nSessionsUsed = nSessionsUsed + 1;

        % Trial-pooled accumulation
        allShortTrials = [allShortTrials; sessShortTrials];
        allLongTrials  = [allLongTrials;  sessLongTrials];
        allShortPuff   = [allShortPuff;   sessShortPuff];
        allLongPuff    = [allLongPuff;    sessLongPuff];

        % Session-pooled accumulation
        if ~isempty(sessShortTrials)
            shortSessionMeans = [shortSessionMeans; mean(sessShortTrials,1,'omitnan')];
            shortSessionNs    = [shortSessionNs;   size(sessShortTrials,1)];
            shortSessionPuff  = [shortSessionPuff; median(sessShortPuff,'omitnan')];
        end
        if ~isempty(sessLongTrials)
            longSessionMeans  = [longSessionMeans;  mean(sessLongTrials,1,'omitnan')];
            longSessionNs     = [longSessionNs;     size(sessLongTrials,1)];
            longSessionPuff   = [longSessionPuff;   median(sessLongPuff,'omitnan')];
        end
    end

    pooled = struct();
    pooled.nSessionsUsed = nSessionsUsed;

    switch lower(cfg.poolMode)
        case 'trials'
            pooled.short = packFromTrials(tGrid, allShortTrials, allShortPuff);
            pooled.long  = packFromTrials(tGrid, allLongTrials,  allLongPuff);

        case 'sessions'
            pooled.short = packFromSessionMeans(tGrid, shortSessionMeans, shortSessionPuff, shortSessionNs);
            pooled.long  = packFromSessionMeans(tGrid, longSessionMeans,  longSessionPuff,  longSessionNs);

        otherwise
            error('cfg.poolMode must be ''trials'' or ''sessions''.');
    end
end

function [F_short, F_long, puff_short, puff_long] = extractShortLongTrials(SessionData, tGrid, cfg)

    trials = SessionData.RawEvents.Trial;
    nTrials = numel(trials);

    F_short = [];
    F_long  = [];
    puff_short = [];
    puff_long  = [];

    for k = 1:nTrials
        tr = trials{1,k};

        % Exclude timeouts
        if cfg.excludeTimeout
            if isfield(tr,'States') && isfield(tr.States,'CheckEyeOpenTimeout')
                v = tr.States.CheckEyeOpenTimeout;
                if ~all(isnan(v))
                    continue;
                end
            end
        end

        % Exclude probes
        if cfg.excludeProbe
            if isfield(tr,'Data') && isfield(tr.Data,'IsProbeTrial') && tr.Data.IsProbeTrial == 1
                continue;
            end
        end

        % Required fields
        if ~isfield(tr,'Data') || ~isfield(tr.Data,'FEC') || ~isfield(tr.Data,'FECTimes')
            continue;
        end
        if ~isfield(tr,'Events') || ~isfield(tr.Events,'GlobalTimer1_Start') || isempty(tr.Events.GlobalTimer1_Start)
            continue;
        end
        if ~isfield(tr,'Events') || ~isfield(tr.Events,'GlobalTimer2_Start') || isempty(tr.Events.GlobalTimer2_Start)
            continue;
        end
        if ~isfield(tr,'States') || ~isfield(tr.States,'LED_Puff_ISI') || isempty(tr.States.LED_Puff_ISI)
            continue;
        end

        FEC  = tr.Data.FEC(:);
        tAbs = tr.Data.FECTimes(:);
        if isempty(FEC) || numel(FEC) ~= numel(tAbs)
            continue;
        end

        tLED  = tr.Events.GlobalTimer1_Start(1);
        tPuff = tr.Events.GlobalTimer2_Start(1);

        % Align time to LED
        tRel = tAbs - tLED;

        % Smooth (optional)
        if cfg.smoothWin > 1
            FEC = movmean(FEC, cfg.smoothWin, 'omitnan');
        end

        % Resample to common grid
        Fq = interp1(tRel, FEC, tGrid, cfg.interpMethod, nan);

        % Classify short/long by ISI duration
        isiDur = tr.States.LED_Puff_ISI(2) - tr.States.LED_Puff_ISI(1);
        puffRel = tPuff - tLED;

        if isiDur <= cfg.shortISI_max
            F_short = [F_short; Fq];
            puff_short = [puff_short; puffRel];
        else
            F_long = [F_long; Fq];
            puff_long = [puff_long; puffRel];
        end
    end
end

function out = packFromTrials(tGrid, Fmat, puffTimes)
    out = struct();
    out.t = tGrid;

    if isempty(Fmat)
        out.mean = nan(size(tGrid));
        out.sem  = nan(size(tGrid));
        out.nTrials = 0;
        out.airpuff_t = nan;
        return;
    end

    out.nTrials = size(Fmat,1);
    out.mean = mean(Fmat, 1, 'omitnan');
    out.sem  = std(Fmat, 0, 1, 'omitnan') ./ sqrt(out.nTrials);

    if isempty(puffTimes)
        out.airpuff_t = nan;
    else
        out.airpuff_t = median(puffTimes, 'omitnan');
    end
end

function out = packFromSessionMeans(tGrid, sessionMeans, puffTimesPerSession, sessionNs)
    out = struct();
    out.t = tGrid;

    if isempty(sessionMeans)
        out.mean = nan(size(tGrid));
        out.sem  = nan(size(tGrid));
        out.nTrials = 0;
        out.airpuff_t = nan;
        return;
    end

    % Each row is a session mean trace
    out.mean = mean(sessionMeans, 1, 'omitnan');
    out.sem  = std(sessionMeans, 0, 1, 'omitnan') ./ sqrt(size(sessionMeans,1));

    % Report total trials contributing (for your bookkeeping)
    if nargin >= 4 && ~isempty(sessionNs)
        out.nTrials = sum(sessionNs);
    else
        out.nTrials = NaN;
    end

    if isempty(puffTimesPerSession)
        out.airpuff_t = nan;
    else
        out.airpuff_t = median(puffTimesPerSession, 'omitnan');
    end
end