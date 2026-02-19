
function plot_probe_data(common_time_vector, data_matrix, color, label, N_SD_1 ,N_SD_2 ,N_C,firstSessionDate,lastSessionDate)
    if isempty(data_matrix)
        return;
    end
    avg = mean(data_matrix, 1, 'omitnan');

    plot(common_time_vector, avg, 'Color', color, 'DisplayName', label, 'LineWidth', 1.5);

    % legend({...
    % ['SD+1 Session Trials (n = ', num2str(N_SD_1), ')'], ...
    % ['SD+2 Session Trials (n = ', num2str(N_SD_2), ')'], ...
    % sprintf(['Control Session Trials (n = %d)\n(%s to %s)'], ...
    %         N_C, ...
    %         datestr(firstSessionDate, 'mm/dd/yyyy'), ...
    %         datestr(lastSessionDate, 'mm/dd/yyyy'))}, ...
    % 'Location', 'bestoutside', ...
    % 'Interpreter', 'latex', ...
    % 'FontName', 'Times New Roman', ...
    % 'FontSize', 10, ...
    % 'Box', 'off');

    sem = std(data_matrix, 0, 1, 'omitnan') ./ sqrt(size(data_matrix, 1));  % Corrected here
    fill([common_time_vector, fliplr(common_time_vector)], ...
         [avg + sem, fliplr(avg - sem)], ...
         color, 'EdgeColor', 'none', 'FaceAlpha', 0.3);




    % Add LED onset shading
    y_fill = [-0.2 -0.2 1 1];
    x_fill_shortLED = [0 0.05 0.05 0];
    fill(x_fill_shortLED, y_fill, [0.5 0.5 0.5], 'FaceAlpha', 0.16, 'EdgeColor', 'none','HandleVisibility', 'off');

xlim([-0.2 0.6]);
ylim([0 1]);
ylabel('Eyelid closure (norm)', 'interpreter', 'latex', 'fontsize', 12);
set(gca, 'TickDir', 'out');




end
