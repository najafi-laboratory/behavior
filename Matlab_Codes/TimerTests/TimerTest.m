timerPing = timer;
timerPong = timer;

timerPing.TimerFcn = @(~,~)pingPongActivity(true,  timerPing, timerPong);
timerPing.Name = 'PingTimer';

timerPong.TimerFcn = @(~,~)pingPongActivity(false, timerPing, timerPong);
timerPong.Name = 'PongTimer';

timerPing.StartDelay = 0.001;
start(timerPing);

%t = timerfind; stop(t); delete(t)

function pingPongActivity(isPing, timerPing, timerPong)
    if isPing
        disp(['PING (' datestr(now,'yyyy-mm-dd HH:MM:SS.FFF') ')'])
    else
        disp(['PONG (' datestr(now,'yyyy-mm-dd HH:MM:SS.FFF') ')'])
    end
    delayTime = ceil(rand*10);
    display(['    delaying '  num2str(delayTime) ' sec.'])
    if isPing
        nextTimer = timerPong;
    else
        nextTimer = timerPing;
    end
    set(nextTimer,'StartDelay', delayTime);
    start(nextTimer);
end