function SoftCodeHandler_EBC(code)
% global M
global S
global BpodSystem
global MEV

switch true
    case code == 1
        MEV.triggerTrialsVideo();
        SendBpodSoftCode(1); % Indicate to the state machine that video start
    case code == 2
        % t = timerfindall;  % returns all timers (running or stopped)
        % disp(t)

        % t = timerfindall('Running','on');
        % fprintf('Running timers: %d\n', numel(t));
        % 
        % for k = 1:numel(t)
        %     fprintf('Timer %d: Name="%s", Mode=%s, Period=%.2f\n', ...
        %         k, t(k).Name, t(k).ExecutionMode, t(k).Period);
        % end



        % fig = gcf;
        % lsn = findall(fig, 'Type', 'listener');
        % fprintf('Number of listeners on figure: %d\n', numel(lsn));
        % for k = 1:numel(lsn)
        %     disp(lsn(k).Callback)     % function handle
        %     disp(lsn(k).EventName)    % which event triggers it
        % end

        % ax = gca;
        % lsn_ax = findall(ax, 'Type', 'listener');
        % fprintf('Number of listeners on axes: %d\n', numel(lsn_ax));

        % 
        % fprintf('Iteration %d: %d timers, %d fig listeners\n', ...
        %         S.GUI.currentTrial, numel(timerfindall), numel(findall(gcf,'Type','listener')));


        MEV.eyeOpen = false;
        ticB = tic;
        % ticA = tic;
        while ~MEV.eyeOpen
            % MEV.checkEyeOpen();
            % tA = toc(ticA);
            % fprintf('tA = %.3f\n', tA);
            CheckEyeOpenTimeCheck = toc(ticB);
            if CheckEyeOpenTimeCheck > S.GUI.CheckEyeOpenTimeout
                SendBpodSoftCode(2);
                break;
            end
            pause(0.10);            
        end
        
        SendBpodSoftCode(1); % Indicate to the state machine that eye is open relative to threshold   
    case code == 3
        MEV.LEDOnsetTime = seconds(datetime("now") - MEV.trialVidStartTime);
        MEV.plotLEDOnset;
    case code == 4
        MEV.AirPuffOnsetTime = seconds(datetime("now") - MEV.trialVidStartTime);
        MEV.plotAirPuffOnset;
    case code >= 20 && code <= 30
        BpodSystem.PluginObjects.V.play(code);
    case code == 255
        BpodSystem.PluginObjects.V.stop;
end
end

% function steps = degrees2MotorSteps(degrees, nMotorStepsPerRev)
%     steps = round((degrees/360)*nMotorStepsPerRev);
% end
% 
% function SetMotorPos = ConvertMaestroPos(MaestroPosition)
%     m = 0.002;
%     b = -3;
%     SetMotorPos = MaestroPosition * m + b;
% end