        %% Set the state matrix
        
        if strcmp(value(rewardStage), 'Direct')
            randomState = 'direct_water';
        else
            randomState = 'wait_for_decision';
        end
            
        startTone = SoundManagerSection(obj, 'get_sound_id', 'WaitStart'); % trial-initiation tone
        flickStim = SoundManagerSection(obj, 'get_sound_id', 'TargetSound'); % flickering stimulus, the main stimulus (visual/auditory)
        goTone = SoundManagerSection(obj, 'get_sound_id', 'WaitEnd'); % go tone
        incorrectTone = SoundManagerSection(obj, 'get_sound_id', pnoise_name); % incorrect-choice tone
        
        sma = StateMachineAssembler('full_trial_structure');
        
        sma = add_scheduled_wave(sma, 'name', 'center_unconst', 'preamble', randi(value(CenterPoke_when)));
        
        trial_ports = ardttl1 + scopeTTL;
        
        %% ITI 
        % in your new code (with the start_rotary_scope state), mouse will have to stop licking for 500ms
        % in order to get the start tone. Then it seems the iti, iti2
        % states and preventing the mouse from licking at those states
        % could be redundant.... maybe you can remove the no-lick
        % constraint during iti, iti2 states.
        
        % iti : so ITI indicates ITI from previous trial
%         sma = add_state(sma, 'name', 'iti',...
%             'self_timer', max(1E-6, value(ITI)-1),... % you are subtracting 1sec, bc you are adding 500ms to the ITI right before the start tone, and 500ms right after the final state. However you are preventing the mouse from lick only during the first 500ms, so in this new code the mouse is forced for a shorter amount of time not to lick during iti.
%             'input_to_statechange', {'Tup','start_rotary_scope', centerLick,'iti2', correctLick,'iti2', errorLick,'iti2'});        
        
        sma = add_state(sma, 'name', 'iti',...
            'self_timer', max(1E-6, value(ITI)-1),... % you are subtracting 1sec, bc you are adding 500ms to the ITI right before the start tone, and 500ms right after the final state. However you are preventing the mouse from lick only during the first 500ms, so in this new code the mouse is forced for a shorter amount of time not to lick during iti.
            'input_to_statechange', {'Tup','trial_start_rot_scope', correctLick,'iti2', errorLick,'iti2'});
        
        
        % iti2
%         sma = add_state(sma, 'name', 'iti2',...
%             'self_timer', max(1E-6, value(ITI)-1),...        
%             'input_to_statechange', {'Tup','start_rotary_scope', centerLick,'iti', correctLick,'iti', errorLick,'iti'});    
        
        sma = add_state(sma, 'name', 'iti2',...
            'self_timer', max(1E-6, value(ITI)-1),...        
            'input_to_statechange', {'Tup','trial_start_rot_scope', correctLick,'iti', errorLick,'iti'});
        
        
        %% Start acquiring mscan data and send the trialStart signal (also the rotary signal). 
        % I think you should have the following asap to make the length of
        % trialStart constant in the analogue channel and make computatoins
        % easier!
        %
        sma = add_state(sma, 'name', 'trial_start_rot_scope',...
            'self_timer', 0.036,...        
            'output_actions', {'DOut', trial_ports + trialStart},...
            'input_to_statechange', {'Tup','start_rotary_scope'});
        
        % remove trialStart from start_rotary_scope. add the name of this
        % stateto Tup iti and iti2.
        %
        
        %% change the name of this state when you start new mice! 
        % The name of this state is misleading now; it is really a short
        % nocenterlick state right before trial initiation.
        
        %%% start_rotary_scope --> scopeTTL gets sent here.
        % you set it to 500ms, bc you want to record imaging and rotary data during some part of the ITI as well.
        sma = add_state(sma, 'name', 'start_rotary_scope',...
            'self_timer', value(startScopeDur),... % 0.035 is the min. Matt: Have your protocol wait at least 35 ms between telling the microscope to start scanning and sending the trial code. The scope is always running the scan mirrors, and waits until the start of the next frame to actually start recording. If you don't do this, you'll lose the start of the codes.
            'output_actions', {'DOut', trial_ports},...
            'input_to_statechange', {'Tup','trialCode1', centerLick,'start_rotary_scope2', correctLick,'start_rotary_scope2', errorLick,'start_rotary_scope2'});
        
        
        sma = add_state(sma, 'name', 'start_rotary_scope2',...
            'self_timer', value(startScopeDur),... 
            'output_actions', {'DOut', trial_ports},...
            'input_to_statechange', {'Tup','trialCode1', centerLick,'start_rotary_scope', correctLick,'start_rotary_scope', errorLick,'start_rotary_scope'});
        
        
        %% (still ITI) Sending the trial code: barcode 2of5 indicates trial number. This will be sent to an AI line on the scope machine.        
        % right now you are allowing the mouse to lick during trialCode time (even though he is in the ITI). 
        % If you dont want to allow it, add the following code to input_to_statechange below. (but it will mean that the trial code will be aborted and will be again sent). centerLick,'start_rotary_scope', correctLick,'start_rotary_scope', errorLick,'start_rotary_scope'
        
        % trialCode1 :  Right here, before the start-tone, is when the trial code gets sent.
        
%         durCode = 0;
        stateNum = 0;
        for pair = 1:size(code, 2)
            stateNum = stateNum + 1;
            stateName = ['trialCode' num2str(stateNum)];
            nextState = [stateName 'Low'];
            
            % High state (send bars, ie send scopeTrial pulse). Always
            % followed by low state.
            sma = add_state(sma, 'name', stateName, ...
                'self_timer', codeModuleDurs(code(1, pair)), ...
                'output_actions', {'DOut', trial_ports + scopeTrial}, ...
                'input_to_statechange', {'Tup', nextState}); % right now you are allowing the mouse to lick during trialCode time (even though he is in the ITI). If you dont want to allow it, add the following, but it will mean that the trial code will be aborted and will be again sent. centerLick,'start_rotary_scope', correctLick,'start_rotary_scope', errorLick,'start_rotary_scope'
            
            
            % Low state (send spaces, ie dont send scopeTrial pulse).
            % Either followed by high state or (when the code is all sent) by
            % state wait_for_initiation.
            stateName = nextState;
            if pair == size(code, 2)
                nextState = 'wait_for_initiation';
            else
                nextState = ['trialCode' num2str(stateNum + 1)];
            end
            sma = add_state(sma, 'name', stateName, ...
                'self_timer', codeModuleDurs(code(2, pair)), ...
                'output_actions', {'DOut', trial_ports}, ...
                'input_to_statechange', {'Tup', nextState}); % right now you are allowing the mouse to lick during trialCode time (even though he is in the ITI). If you dont want to allow it, add the following, but it will mean that the trial code will be aborted and will be again sent. centerLick,'start_rotary_scope', correctLick,'start_rotary_scope', errorLick,'start_rotary_scope'
            
%             durCode = durCode + sum(codeModuleDurs(code(:, pair)));
        end
        

        %% Playing the start tone and waiting for the mouse to initiate the trial.

        % wait_for_initiation
        sma = add_state(sma, 'name', 'wait_for_initiation',...
            'self_timer', value(TimeToInitiate), ... % 60 sec value(TimeToInitiate)
            'output_actions', {'DOut', trial_ports, 'SoundOut', startTone},...
            'input_to_statechange', {'Tup','wait_for_initiation2', ... %used to be iti but i want to keep record of it.
            centerLick,'stim_delay', correctLick,'wrong_initiation', errorLick,'wrong_initiation'}); % wrong_initiation simply leads to state iti.
        
        % wait_for_initiation2
        sma = add_state(sma, 'name', 'wait_for_initiation2',...
            'self_timer', value(TimeToInitiate), ... % 60 sec value(TimeToInitiate)
            'output_actions', {'DOut', trial_ports, 'SoundOut', startTone},...
            'input_to_statechange', {'Tup','wait_for_initiation', ... %used to be iti but i want to keep record of it.
            centerLick,'stim_delay', correctLick,'wrong_initiation', errorLick,'wrong_initiation'}); % wrong_initiation simply leads to state iti.
        
            
%{            
        % wait_for_initiation        
        if ~value(CenterPoke_amount)
            % wait_for_initiation
            sma = add_state(sma, 'name', 'wait_for_initiation',...
                'self_timer', value(TimeToInitiate), ... % 60 sec value(TimeToInitiate)
                'output_actions', {'DOut', trial_ports, 'SoundOut', startTone, 'SchedWaveTrig', 'center_unconst'},...
                'input_to_statechange', {'Tup','wait_for_initiation2', ... %used to be iti but i want to keep record of it.
                centerLick,'stop_center_unconst', correctLick,'wrong_initiation', errorLick,'wrong_initiation'}); % wrong_initiation simply leads to state iti.
            
            % wait_for_initiation2
            sma = add_state(sma, 'name', 'wait_for_initiation2',...
                'self_timer', value(TimeToInitiate), ... % 60 sec value(TimeToInitiate)
                'output_actions', {'DOut', trial_ports, 'SoundOut', startTone},...
                'input_to_statechange', {'Tup','wait_for_initiation', ... %used to be iti but i want to keep record of it.
                centerLick,'stop_center_unconst', correctLick,'wrong_initiation', errorLick,'wrong_initiation'}); % wrong_initiation simply leads to state iti.
            
        else % give a large drop of water in the center if the mouse did not initiate a trial in [180 300] seconds. water amount = water_duration_ave*value(CenterPoke_amount)
            
            % wait_for_initiation
            sma = add_state(sma, 'name', 'wait_for_initiation',...
                'self_timer', value(TimeToInitiate), ... % 60 sec value(TimeToInitiate)
                'output_actions', {'DOut', trial_ports, 'SoundOut', startTone, 'SchedWaveTrig', 'center_unconst'},...
                'input_to_statechange', {'Tup','wait_for_initiation2', ... %used to be iti but i want to keep record of it.
                centerLick,'stop_center_unconst', correctLick,'wrong_initiation', errorLick,'wrong_initiation',...
                'center_unconst_In','give_center_unconst'}); % wrong_initiation simply leads to state iti.
            
            % wait_for_initiation2
            sma = add_state(sma, 'name', 'wait_for_initiation2',...
                'self_timer', value(TimeToInitiate), ... % 60 sec value(TimeToInitiate)
                'output_actions', {'DOut', trial_ports, 'SoundOut', startTone},...
                'input_to_statechange', {'Tup','wait_for_initiation', ... %used to be iti but i want to keep record of it.
                centerLick,'stop_center_unconst', correctLick,'wrong_initiation', errorLick,'wrong_initiation',...
                'center_unconst_In','give_center_unconst'}); % wrong_initiation simply leads to state iti.

        end        
        
        % give_center_unconst: % give a large drop of water in the center if the mouse did not initiate a trial in [180 300] seconds. water amount = water_duration_ave*value(CenterPoke_amount)
        sma = add_state(sma, 'name', 'give_center_unconst',...
            'self_timer', water_duration_ave*value(CenterPoke_amount), ...
            'output_actions', {'DOut', trial_ports + cwater}, ...
            'input_to_statechange', {'Tup','wait_for_initiation', ... %used to be iti but i want to keep record of it.
            centerLick,'stop_center_unconst', correctLick,'wrong_initiation', errorLick,'wrong_initiation'});              
        
        
        
        % stop_center_unconst (stop the center_unconst scheduled wave)
        sma = add_state(sma, 'name', 'stop_center_unconst',...
            'self_timer', 1E-6,...
            'output_actions', {'DOut', trial_ports, 'SchedWaveTrig', '-center_unconst'},...
            'input_to_statechange', {'Tup','stim_delay'});
%}            
        
        %% Mouse initiating the trial and stimulus playing.
        % stim_delay % 
        sma = add_state(sma, 'name', 'stim_delay',...
            'self_timer', value(PreStimDelay),...
            'output_actions', {'DOut', trial_ports},...
            'input_to_statechange', {'Tup','wait_stim', correctLick,'early_decision', errorLick,'early_decision'}); % early_decision is just like iti. Defined to keep trach of early decisions.
        
        % wait_stim; in this state stim will start playing; its duration is
        % waitDur + extraStimDur + stimDur_aftRew; if stimDur_diff is
        % non-zero, then stim duration will be simDur_diff +
        % extraStimDur + stimDur_aftRew
        sma = add_state(sma, 'name', 'wait_stim',...
            'self_timer', wait_dur + value(PostStimDelay),...
            'output_actions', {'DOut', trial_ports, 'SoundOut', flickStim},...
            'input_to_statechange', {'Tup','center_reward', correctLick,'early_decision0', errorLick,'early_decision0'}); %randomState
  
        %{
        % the following 2 states were added in fn_lick_1_3
        % lickcenter_again
        sma = add_state(sma, 'name', 'lickcenter_again',...
            'self_timer', value(CenterAgain), ... % .3 you may want to make this shorter as the training gets better.
            'output_actions', {'DOut', trial_ports},...
            'input_to_statechange', {centerLick, 'center_reward', correctLick,'early_decision0', errorLick,'early_decision0', 'Tup','did_not_lickagain'});
        %}
        
        % center_reward. 
        % randomSate is direct_water for Direct, and wait_for_decision for Allow correction and Choose side.
        sma = add_state(sma, 'name', 'center_reward', ...
            'self_timer', 1E-6,... % value(CenterWater)
            'output_actions', {'DOut', cwater + trial_ports, 'SoundOut',goTone},...
            'input_to_statechange', {'Tup', randomState});
        
        
        %% Mouse choosing a side spout.
        % states below come after the mouse correctly initiates a trial (ie he does 1st lick and commit lick and receives center reward). 
 
        % This state will come after center_reward if in Direct stage (so randomState is direct_water).
        sma  = add_state(sma, 'name', 'direct_water',...
                'self_timer', water_duration,...
                'output_actions', {'DOut', water_delivery + trial_ports},...
                'input_to_statechange', {'Tup', 'wait_for_decision'});       
            
            
        % This state will come after direct_water (for Direct stage).
        if strcmp(value(rewardStage), 'Direct')
            sma = add_state(sma, 'name', 'wait_for_decision',...
            'self_timer', value(TimeToChoose)+3000,...
            'output_actions', {'DOut', trial_ports},...
            'input_to_statechange', {'Tup','did_not_choose', correctLick,'direct_correct'});            
        
        
        
        % This state will come after center_reward if in Allow correction and Choose side (so random state is wait_for_decision).
        else 
            % wait_for_decision. Define correct and incorrect choice as licking for .2sec
            sma = add_state(sma, 'name', 'wait_for_decision',...
                'self_timer', value(TimeToChoose),... +3000
                'output_actions', {'DOut', trial_ports},...
                'input_to_statechange', {'Tup','did_not_choose', correctLick,'correctlick_again_wait', errorLick,'errorlick_again_wait'});
            
            % correctlick_again_wait. mouse will not receive reward unless he licks again on the correct side after .2
            sma = add_state(sma, 'name', 'correctlick_again_wait',...
                'self_timer', sidelick_dur,... % perhaps gradually increase this value to make the choices more costly and less impulsive.
                'output_actions', {'DOut', trial_ports},...
                'input_to_statechange', {'Tup','correctlick_again', errorLick,'errorlick_again_wait'});
                           

            % errorlick_again_wait. if he errorlicks, he will go to a again_wait state (.2), if he licks again after that, now it is actually an errorlick. so state punish_allowcorrect will happen.
            sma = add_state(sma, 'name', 'errorlick_again_wait',...
                'self_timer', sidelick_dur,...
                'output_actions', {'DOut', trial_ports},...
                'input_to_statechange', {'Tup','errorlick_again', correctLick,'correctlick_again_wait'});
            
            
            % correctlick_again. he has a window of .2 to lick again to receive reward.  perhaps later change Tup to 'did_not_sidelickagain', or define a schedulewave. (fn attention) % you changed to did_not_choose on 3/19/15. it used to be 'wait_for_decision'
            sma = add_state(sma, 'name', 'correctlick_again',...
                'self_timer', value(SideAgain),... % .4; .3
                'output_actions', {'DOut', trial_ports},...
                'input_to_statechange', {'Tup','wait_for_decision2', correctLick, 'reward', errorLick,'errorlick_again_wait'});
                        
            
            % errorlick_again. he has a window of .2 to lick again, if he does punishment happens. perhaps later change Tup to 'did_not_choose'
            if strcmp(value(rewardStage), 'Allow correction')
                
                sma = add_state(sma, 'name', 'errorlick_again',...
                    'self_timer', value(SideAgain),...
                    'output_actions', {'DOut', trial_ports},...
                    'input_to_statechange', {'Tup','wait_for_decision2', errorLick, 'punish_allowcorrection', correctLick,'correctlick_again_wait'});
            
            elseif strcmp(value(rewardStage), 'Choose side')
                
                sma = add_state(sma, 'name', 'errorlick_again',...
                    'self_timer', value(SideAgain),...
                    'output_actions', {'DOut', trial_ports},...
                    'input_to_statechange', {'Tup','wait_for_decision2', errorLick, 'punish', correctLick,'correctlick_again_wait'});
            end
            
            
            % wait_for_decision2 : if the mouse didn't lickAgain, give him a second chance to respond properly (ie licking again after sideLickDur).
            sma = add_state(sma, 'name', 'wait_for_decision2',...
                'self_timer', value(TimeToChoose2),... % 4
                'output_actions', {'DOut', trial_ports},...
                'input_to_statechange', {'Tup','did_not_sidelickagain', correctLick,'correctlick_again_wait', errorLick,'errorlick_again_wait'});           
            
        end
        
        %{
        elseif strcmp(value(rewardStage), 'Allow correction')
%             sma = add_state(sma, 'name', 'wait_for_decision',...
%                 'self_timer', value(TimeToChoose),... +3000
%                 'input_to_statechange', {'Tup','did_not_choose', correctLick,'reward'}); % did_not_choose is same as stopSound, but I am defining it so I can have a record of did_not_chooses.
            
            sma = add_state(sma, 'name', 'wait_for_decision',...
                'self_timer', value(TimeToChoose),... +3000
                'input_to_statechange', {'Tup','did_not_choose', correctLick,'reward', errorLick,'punish_allowcorrection'}); % did_not_choose is same as stopSound, but I am defining it so I can have a record of did_not_chooses.
            
            % punishment if allow correction
            sma = add_state(sma, 'name', 'punish_allowcorrection',...
                'self_timer', 1E-6,...
                'output_actions', {'SoundOut', incorrectTone},...            
                'input_to_statechange', {'Tup','wait_for_decision', correctLick,'reward'});
            
            
        elseif strcmp(value(rewardStage), 'Choose side')
            sma = add_state(sma, 'name', 'wait_for_decision',...
                'self_timer', value(TimeToChoose),...
                'input_to_statechange', {'Tup','did_not_choose', correctLick,'reward', errorLick,'punish'});
        end
        %}
        
        
        %% states defining the outcome of a trial.
        
        % wrong_initiation
        sma = add_state(sma, 'name', 'wrong_initiation',...
            'self_timer', 1E-6,...
            'output_actions', {'DOut', trial_ports, 'SchedWaveTrig', '-center_unconst'},...
            'input_to_statechange', {'Tup','stop_rotary_scope'});
        
        
        
        % early_decision0
        sma = add_state(sma, 'name', 'early_decision0',... % like early_decision, but also stops the stimulus. 
            'self_timer', 1E-6,...
            'output_actions', {'DOut', trial_ports, 'SoundOut', -flickStim},...
            'input_to_statechange', {'Tup','early_decision'});        
        
        % early_decision
        sma = add_state(sma, 'name', 'early_decision',... 
            'self_timer', 1E-6,...
            'output_actions', {'DOut', trial_ports},...
            'input_to_statechange', {'Tup','stop_rotary_scope'});
        
        
        
        
        % did_not_lickagain
        sma = add_state(sma, 'name', 'did_not_lickagain',... % like early_decision, but also stops the stimulus. 
            'self_timer', 1E-6,...
            'output_actions', {'DOut', trial_ports, 'SoundOut', -flickStim},...
            'input_to_statechange', {'Tup','stop_rotary_scope'});
        
        
        
        
        % did_not_choose
        sma = add_state(sma, 'name', 'did_not_choose',... 
            'self_timer', 1E-6,...
            'output_actions', {'DOut', trial_ports, 'SoundOut', -flickStim},...
            'input_to_statechange', {'Tup','stop_rotary_scope'});

        
        
        % did_not_sidelickagain
        sma = add_state(sma, 'name', 'did_not_sidelickagain',... 
            'self_timer', 1E-6,...
            'output_actions', {'DOut', trial_ports, 'SoundOut', -flickStim},...
            'input_to_statechange', {'Tup','stop_rotary_scope'});
        
        
        
        % direct_correct (when the animal correctLicks in the Direct stage).  just like did_not_choose.
        sma = add_state(sma, 'name', 'direct_correct',...
            'self_timer', 1E-6,...
            'output_actions', {'DOut', trial_ports, 'SoundOut', -flickStim},...
            'input_to_statechange', {'Tup','stop_rotary_scope'});
        
        
        
        
        % reward
%         sma = add_state(sma, 'name', 'reward',...
%             'self_timer', water_duration,...
%             'output_actions', {'SoundOut', -flickStim, 'DOut', water_delivery},...
%             'input_to_statechange', {'Tup','iti'});
        
        % reward: remember in this new ways, ITI if computed from the time of reward to the start of the next trial will be 1 sec longer that value(ITI). also remember that ITI does not include the duration of check_next_trial_ready, which is about 1.5 sec for my protocol.
        sma = add_state(sma, 'name', 'reward',...
            'self_timer', water_duration,...
            'output_actions', {'DOut', water_delivery + trial_ports, 'SoundOut', -incorrectTone},...
            'input_to_statechange', {'Tup','stopstim_pre'});
        
        % keep playing the stim for StimDur_aftRew sec. after that or if the animal errorLicks the stim stops.
        % but remember if the stim has already stopped, then this state
        % wont mean anything!
        sma = add_state(sma, 'name', 'stopstim_pre',...
            'self_timer', stimdur_aftrew,... % 1
            'output_actions', {'DOut', trial_ports},...
            'input_to_statechange', {'Tup','reward_stopstim', errorLick,'reward_stopstim'});
        
        sma = add_state(sma, 'name', 'reward_stopstim',...
            'self_timer', 1E-6,...
            'output_actions',  {'DOut', trial_ports, 'SoundOut', -flickStim},...
            'input_to_statechange', {'Tup','stop_rotary_scope'});        
        
        
        

        % punish_allowcorrection (consider adding time out too)
        sma = add_state(sma, 'name', 'punish_allowcorrection',...
            'self_timer', value(TimeToChoose2),... % he has timeToChoose sec to correct lick.
            'output_actions', {'DOut', trial_ports, 'SoundOut', incorrectTone},...            
            'input_to_statechange', {'Tup','punish_allowcorrection_done', correctLick,'correctlick_again_wait'});
        
        sma = add_state(sma, 'name', 'punish_allowcorrection_done',... 
            'self_timer', 1E-6,...
            'output_actions', {'DOut', trial_ports, 'SoundOut', -flickStim},...
            'input_to_statechange', {'Tup','stop_rotary_scope'});
        

        % punish (if choose side)
        sma = add_state(sma, 'name', 'punish',...
            'self_timer', 1E-6,...
            'output_actions', {'DOut', trial_ports, 'SoundOut', -flickStim},...
            'input_to_statechange', {'Tup','punish_timeout'});

        % Auditory feedback and Time out punish 
        sma = add_state(sma, 'name', 'punish_timeout',...
            'self_timer', value(errorTimeout),...
            'output_actions', {'DOut', trial_ports, 'SoundOut', incorrectTone},...
            'input_to_statechange', {'Tup','stop_rotary_scope'});
                
         
        
        %% end of states. (another 500ms of the ITI)
        %%% stop_rotary_scope --> last state that includes scopeTTL and
        %%% ardttl. you set it to 500ms, bc you want to record
        %%% imaging and rotary data during ITI.
        sma = add_state(sma, 'name', 'stop_rotary_scope',...
            'self_timer', value(stopScopeDur),...
            'output_actions', {'DOut', trial_ports},...
            'input_to_statechange', {'Tup','check_next_trial_ready'});
        
        
%         % last_state
%         sma = add_state(sma, 'name', 'last_state',...
%             'self_timer', 1E-6,...
%             'input_to_statechange', {'Tup','check_next_trial_ready'});
        
        
        % parsedEvents.states.ending_state will indicate one of the states below which were sent to the assembler.
%         dispatcher('send_assembler', sma, {'punish_timeout', 'punish_allowcorrection_done', 'reward_stopstim', 'direct_correct', 'did_not_choose', 'did_not_sidelickagain', 'did_not_lickagain', 'early_decision', 'wrong_initiation'});
        dispatcher('send_assembler', sma, {'stop_rotary_scope'});
%         dispatcher('send_assembler', sma, {'last_state'});
        
