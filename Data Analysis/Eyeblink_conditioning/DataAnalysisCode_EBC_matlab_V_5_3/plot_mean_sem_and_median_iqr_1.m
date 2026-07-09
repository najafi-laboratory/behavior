function plot_mean_sem_and_median_iqr_1(tvec, X, ttl, varargin)
% plot_mean_sem_and_median_iqr
% Mean ± SEM plot with optional LED and Airpuff shading
%
% Optional name-value pairs:
%   'LED'       : [tStart tEnd]
%   'Puff'      : [tStart tEnd]
%   'PuffColor' : RGB row vector (default blue)

    hold on;

    % Defaults
    LEDwin     = [NaN NaN];
    Puffwin    = [NaN NaN];
    PuffColor  = [0.5 0.5 1.0];   % default blue

    % Parse inputs
    for k = 1:2:numel(varargin)
        switch lower(varargin{k})
            case 'led'
                LEDwin = varargin{k+1};
            case 'puff'
                Puffwin = varargin{k+1};
            case 'puffcolor'
                PuffColor = varargin{k+1};
            otherwise
                error('Unknown option: %s', varargin{k});
        end
    end

    if isempty(X) || all(isnan(X(:)))
        title([ttl ' (no data)']);
        return;
    end

    % ---- Mean ± SEM
    mu  = mean(X,1,'omitnan');
    sd  = std(X,0,1,'omitnan');
    n   = sum(~isnan(X),1);
    sem = sd ./ sqrt(max(n,1));

    hMeanBand = fill([tvec fliplr(tvec)], [mu-sem fliplr(mu+sem)], ...
        [0.8 0.8 0.8], 'EdgeColor','none', 'FaceAlpha',0.6);

    hMean = plot(tvec, mu, 'k', 'LineWidth', 2.5);

    % xline(0,'--');   % LED onset

    % ---- Full-height patches
    yl = ylim;
    y_fill = [yl(1) yl(1) yl(2) yl(2)];

    % ---- LED shading (gray)
    if all(isfinite(LEDwin)) && LEDwin(2) > LEDwin(1)
        fill([LEDwin(1) LEDwin(2) LEDwin(2) LEDwin(1)], y_fill, ...
            [0.5 0.5 0.5], 'FaceAlpha',0.16, 'EdgeColor','none');
    end

    % ---- Airpuff shading (blue or green)
    if all(isfinite(Puffwin)) && Puffwin(2) > Puffwin(1)
        fill([Puffwin(1) Puffwin(2) Puffwin(2) Puffwin(1)], y_fill, ...
            PuffColor, 'FaceAlpha',0.25, 'EdgeColor','none');
        % xline(Puffwin(1),'--');
    end

    xlabel('Time from LED (s)');
    ylabel('Angular speed (d\theta/dt)');
    title(ttl);
    box off;
    set(gca,'TickDir','out');

    legend([hMeanBand hMean], {'Mean \pm SEM','Mean'}, 'Location','best');
end