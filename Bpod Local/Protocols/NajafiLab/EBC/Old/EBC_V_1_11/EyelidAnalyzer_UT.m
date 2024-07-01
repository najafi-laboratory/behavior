MEV = EyelidAnalyzer;


MEV.startGUIVideo();


% MEV.stopGUIVideo;
% 
% 
% MEV.connectVideoTrial;
% pause(0.1);
% MEV.startVideoTrial;
% 
% imaqresasdfads
% %MEV.vid
% MEV.vid.FramesAcquired
% MEV.vid.FramesAvailable
% 
% isrunning(MEV.vid)


% Display all valid trigger configurations.
%triggerinfo(MEV.vid)
%MEV.src.TriggerMode
            vid = videoinput('pointgrey', 1, "F7_Raw8_640x512_Mode1");
            src = getselectedsource(vid);
            % src.Exposure = 1.624512e+00;
            src.ExposureMode = "Auto";
            % src.FrameRate = 3.019109e+01;
            % src.Sharpness = 512;
            %src.SharpnessMode = "Auto";
            %src.FrameRateMode = "Auto";
            src.TriggerDelay = 1.13249 * 0.000001;
            src.FrameRateMode = "Off";
            src.FrameRate = 473.6355;
            src.Shutter = 0.01;
            src.Strobe3 = 'On';
            % delete(vid);


            % vid = videoinput('gentl', 1);     
            disp(['Video Input Connected']);
            % Configure Video Source Properties
            src = getselectedsource(vid);

            % check fps of camera
            % fps = src.AcquisitionFrameRate;
            fps = src.FrameRate;

            % fpsCheck = 0;
            % if fpsCheck
            %     if ~((fps > 30) && (fps < 31))
            %         %disp(['Set camera FPS = 30']);
            %         onGUIClose;
            %         MatExc = MException('MyComponent:noSuchVariable','Set camera FPS to 30 fps. fps = %s', num2str(fps));
            %         throw(MatExc);               
            %     end
            % end
            framesPerTrigger = 100;
            % numTriggers = 200;
            numTriggers = 100;
            triggerCondition = "risingEdge";
            triggerSource = "externalTriggerMode0-Source2";

            triggerconfig(vid, "hardware", triggerCondition, triggerSource);
            vid.FramesPerTrigger = framesPerTrigger;
            vid.TriggerRepeat = numTriggers - 1;


            start(vid)
            vid.FramesAvailable

             [data, time, metadata] = getdata(vid, vid.FramesAvailable);


                    timeCheck = time - time(1);
                    Ts = diff(time);
                    avgFps = 1/mean(Ts)
