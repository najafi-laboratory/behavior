clc; clear; close all;

% ------------ sessions ------------
data_files = dir('*_EBC_*.mat');   % adjust pattern if needed

% ------------ settings ------------
t_pre  = 0.2;
t_post = 0.6;
dt     = 1/250;
smoothWin = 5;                    % 0=no smoothing, else movmean samples

excludeProbe   = true;
excludeTimeout = true;

good_CR_threshold = 0.05;
poor_CR_threshold = 0.02;

interpMethod = 'linear';

% Short/Long split
shortISI_max = 0.30;  % <=0.30s short, >0.30s long

% ------------ date range for naming/annotation ------------
dateTok = regexp({data_files.name}, '\d{8}', 'match', 'once');
sessionDates = datetime(dateTok, 'InputFormat','yyyyMMdd');
sessionDates = sessionDates(~isnat(sessionDates));
firstTok = char(min(sessionDates), 'yyyyMMdd');
lastTok  = char(max(sessionDates), 'yyyyMMdd');

firstDate = char(min(sessionDates), 'MM/dd/yyyy');
lastDate  = char(max(sessionDates), 'MM/dd/yyyy');

% ------------ collect + classify ------------
tGrid = -t_pre:dt:t_post;

% Accumulators: {Short, Long} x {Good, Poor, No}
F_short_good = []; puff_short_good = [];
F_short_poor = []; puff_short_poor = [];
F_short_no   = []; puff_short_no   = [];

F_long_good  = []; puff_long_good  = [];
F_long_poor  = []; puff_long_poor  = [];
F_long_no    = []; puff_long_no    = [];

nSessionsUsed = 0;

for i = 1:numel(data_files)
    S = load(data_files(i).name);
    if ~isfield(S,'SessionData'), continue; end
    SD = S.SessionData;
    nSessionsUsed = nSessionsUsed + 1;

    trials = SD.RawEvents.Trial;
    for k = 1:numel(trials)
        tr = trials{1,k};

        % exclude timeout
        if excludeTimeout
            if isfield(tr,'States') && isfield(tr.States,'CheckEyeOpenTimeout')
                v = tr.States.CheckEyeOpenTimeout;
                if ~all(isnan(v)), continue; end
            end
        end

        % exclude probe
        if excludeProbe
            if isfield(tr,'Data') && isfield(tr.Data,'IsProbeTrial') && tr.Data.IsProbeTrial==1
                continue;
            end
        end

        % required fields
        if ~isfield(tr,'Data') || ~isfield(tr.Data,'FEC') || ~isfield(tr.Data,'FECTimes'), continue; end
        if ~isfield(tr,'Events') || ~isfield(tr.Events,'GlobalTimer1_Start') || isempty(tr.Events.GlobalTimer1_Start), continue; end
        if ~isfield(tr,'Events') || ~isfield(tr.Events,'GlobalTimer2_Start') || isempty(tr.Events.GlobalTimer2_Start), continue; end
        if ~isfield(tr,'States') || ~isfield(tr.States,'LED_Puff_ISI') || isempty(tr.States.LED_Puff_ISI), continue; end

        FEC  = tr.Data.FEC(:);
        tAbs = tr.Data.FECTimes(:);
        if isempty(FEC) || numel(FEC) ~= numel(tAbs), continue; end

        tLED_abs  = tr.Events.GlobalTimer1_Start(1);
        tPuff_abs = tr.Events.GlobalTimer2_Start(1);

        % LED-aligned time
        tRel = tAbs - tLED_abs;
        puffRel = tPuff_abs - tLED_abs;

        % smooth before interpolation
        if smoothWin > 1
            FEC = movmean(FEC, smoothWin, 'omitnan');
        end

        % interpolate onto common grid
        Fq = interp1(tRel, FEC, tGrid, interpMethod, nan);

        % short/long by ISI duration
        isiDur = tr.States.LED_Puff_ISI(2) - tr.States.LED_Puff_ISI(1);
        isShort = isiDur <= shortISI_max;

        % classify CR on the common grid (LED at 0, puff at puffRel)
        [CR_category, ~, ~] = classify_CR_05(tGrid, Fq, 0, puffRel, good_CR_threshold, poor_CR_threshold);

        % route to accumulator
        if isShort
            switch CR_category
                case 'Good CR'
                    F_short_good = [F_short_good; Fq];  puff_short_good = [puff_short_good; puffRel];
                case 'Poor CR'
                    F_short_poor = [F_short_poor; Fq];  puff_short_poor = [puff_short_poor; puffRel];
                otherwise
                    F_short_no   = [F_short_no;   Fq];  puff_short_no   = [puff_short_no;   puffRel];
            end
        else
            switch CR_category
                case 'Good CR'
                    F_long_good = [F_long_good; Fq];    puff_long_good = [puff_long_good; puffRel];
                case 'Poor CR'
                    F_long_poor = [F_long_poor; Fq];    puff_long_poor = [puff_long_poor; puffRel];
                otherwise
                    F_long_no   = [F_long_no;   Fq];    puff_long_no   = [puff_long_no;   puffRel];
            end
        end
    end
end

% ------------ pack stats ------------
S_short_good = packStats(tGrid, F_short_good, puff_short_good);
S_short_poor = packStats(tGrid, F_short_poor, puff_short_poor);
S_short_no   = packStats(tGrid, F_short_no,   puff_short_no);

S_long_good  = packStats(tGrid, F_long_good,  puff_long_good);
S_long_poor  = packStats(tGrid, F_long_poor,  puff_long_poor);
S_long_no    = packStats(tGrid, F_long_no,    puff_long_no);

% ------------ plot 3x2 (rows=CR class, cols=Short/Long) ------------
figure('Position',[100 100 1150 850]);
tiledlayout(3,2,'TileSpacing','compact','Padding','compact');

plotPanel(S_short_good, 'Good CR', 'Short');
plotPanel(S_long_good,  'Good CR', 'Long');

plotPanel(S_short_poor, 'Poor CR', 'Short');
plotPanel(S_long_poor,  'Poor CR', 'Long');

plotPanel(S_short_no,   'No CR',   'Short');
plotPanel(S_long_no,    'No CR',   'Long');

% ticks out (all axes)
set(findall(gcf,'Type','axes'), 'TickDir','out');

% bestoutside session range
legend(sprintf('Sessions:\\newline %s--%s\\newline sessions=%d', firstDate, lastDate, nSessionsUsed), ...
       'Location','bestoutside','Interpreter','latex','FontSize',12,'Box','off');

% ------------ save PDF ------------
exportgraphics(gcf, sprintf('PooledAvgFEC_CRclassified_ShortLong_%s_to_%s.pdf', firstTok, lastTok), ...
    'ContentType','vector');


%% -------- local helpers --------
function out = packStats(t, Fmat, puffTimes)
    out.t = t;
    out.n = size(Fmat,1);
    if isempty(Fmat)
        out.mean = nan(size(t));
        out.sem  = nan(size(t));
        out.puff = nan;
    else
        out.mean = mean(Fmat, 1, 'omitnan');
        out.sem  = std(Fmat, 0, 1, 'omitnan') ./ sqrt(out.n);
        out.puff = median(puffTimes, 'omitnan');
    end
end

function plotPanel(S, crLabel, blockLabel)
    nexttile; hold on;

    plot(S.t, S.mean, 'LineWidth', 1.6);
    fill([S.t fliplr(S.t)], [S.mean+S.sem, fliplr(S.mean-S.sem)], ...
        'k', 'FaceAlpha', 0.15, 'EdgeColor','none');

    xline(0,'--','LED','LabelVerticalAlignment','bottom','LabelHorizontalAlignment','left');
    if ~isnan(S.puff)
        xline(S.puff,'--','Airpuff','LabelVerticalAlignment','bottom','LabelHorizontalAlignment','left');
    end

    title(sprintf('%s -- %s (n=%d)', crLabel, blockLabel, S.n), ...
        'Interpreter','latex','FontSize',14);

    xlabel('Time from LED onset (s)');
    ylabel('FEC');
    grid off;
end