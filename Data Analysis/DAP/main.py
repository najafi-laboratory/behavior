# %%
from datetime import datetime
import random
import sys
import argparse
import logging
import pandas as pd


import modules.utils.utils as utils
# import modules.config as config

from modules.core.pipeline_manager import PipelineManager
from modules.utils import run_suite2p_pipeline as s2p


# import modules.data_extract as extract
# import modules.data_load as load_data
# import modules.data_preprocess as preprocess
# import modules.session_extractor as session_extractor


# if __name__ == "__main__":
#     # Get the current date and time
#     current_datetime = datetime.now()
    
#     # Format the date as 'yyyymmdd'
#     formatted_date = current_datetime.strftime('%Y%m%d')
    
#     # random 4-digit number as string
#     random_4_digit_str = f"{random.randint(0, 9999):04d}"
    
#      # subject_list = ['LCHR_TS01', 'LCHR_TS02', 'SCHR_TS06', 'SCHR_TS07', 'SCHR_TS08', 'SCHR_TS09']
#     # subject_list = ['LCHR_TS02', 'SCHR_TS06', 'SCHR_TS07', 'SCHR_TS08', 'SCHR_TS09']
#     subject_list = ["LCHR_MC01", "LCHR_MC02"]
#     # subject_list = ["LCHR_TS02"]
    
#     # subject_list = ["SCHR_TS06"]
#     # subject_list = ["SCHR_TS07", 'SCHR_TS08', 'SCHR_TS09']
    
#     subjects = ", ".join([s for s in subject_list])
#     print(f"[INFO] Extracting data for subjects {subjects}...")

#     selected_config = config.session_config_list_2AFC
    # Instantiate the extractor with subject_list, config, selected_config, force
    # extractor = session_extractor.SessionExtractor(subject_list, config, selected_config, force=False)

    # # Extract and store sessions using the extractor instance
    # extractor.batch_extract_sessions(force=None)
    
    
    
    # pipeline_manager = PipelineManager(subject_list, config, selected_config)
    
    # print(f"Preprocessing data for subjects {subjects}...")    
    
    # # datist - data extraction, data bridging, data cleaning, phantom data, data snoopingecho
    # all_subjects_data = load_data.load_batch_sessions(subject_list, config)
    
    # preprocessed_data = preprocess.batch_preprocess_sessions(subject_list, config, force=True)


    # print(f"Running data analysis for subjects {subjects}...")
    
    
def main():
    """Main function with flexible subject and experiment selection"""

    # Generate run ID first
    current_datetime = datetime.now()
    formatted_date = current_datetime.strftime('%Y%m%d')
    random_4_digit_str = f"{random.randint(0, 9999):04d}"
    run_id = f"{formatted_date}_{random_4_digit_str}"
    
    # Setup logging for entire application
    logger = utils.setup_logging(run_id)
    logger.info("Main: === Data Analysis Pipeline Started ===")

    # Debug flag - set to True to force interactive mode
    FORCE_INTERACTIVE = True  # Set this to True for debugging

    run_interactive_mode(run_id, logger)

    # # For interactive/development use - no command line arguments
    # if len(sys.argv) == 1 or FORCE_INTERACTIVE:
    #     return run_interactive_mode(run_id, logger)

    # # Command line mode with argument parsing
    # return run_command_line_mode(run_id, logger)

def run_interactive_mode(run_id, logger):
    """Interactive mode for development and testing"""
    logger.info("Main: === Data Analysis Pipeline - Interactive Mode ===")

    # === CONFIGURATION SECTION ===
    # Easy to modify for different runs
    
    # Option 1: Specify subjects directly
    # subject_selection = "LCHR_MC01,LCHR_MC02"  # Comma-separated list
    # subject_selection = "LCHR_MC01"  # Comma-separated list
    # subject_selection = "YH24LG"  # Comma-separated list
    subject_selection = "Test"  # Comma-separated list
    
    # Option 2: Use predefined groups (uncomment to use)
    # subject_selection = "lchr_subjects"        # From subject_groups in YAML
    # subject_selection = "all_2AFC"             # From subject_groups in YAML
    
    # Option 3: Use all subjects from an experiment (uncomment to use)
    # subject_selection = "session_config_list_2AFC"  # All subjects in this experiment
    
    # Option 4: Use None to get all subjects for the experiment (uncomment to use)
    # subject_selection = None
    
    # Experiment configuration to use
    # experiment_config = "single_interval_discrimination"
    experiment_config = "eye_blink_conditioning"
    
    # Pipeline control flags
    skip_extraction = False
    skip_preprocessing = False
    skip_loading = False
    skip_analysis = False
    skip_visualization = False
    skip_reporting = False
    
    # skip_extraction = True
    # # skip_preprocessing = True
    # skip_loading = True
    # skip_analysis = True
    # skip_visualization = True
    # skip_reporting = True
    
    force_extraction = False
    force_preprocessing = True    
    
    # force_extraction = False
    # force_preprocessing = False
    
    # Development flags
    interactive_plots = True  # Show plots during sdevelopment
    spyder_mode = True  # Set to True when running in Spyder
    
    
    # === END CONFIGURATION SECTION ===
    
    # Initialize pipeline with config, experiment, and optional subjects
    pipeline = PipelineManager(
        config_path='modules/config/config.yaml',
        experiment_config=experiment_config,
        subject_selection=subject_selection,
        run_id=run_id,  # Pass run_id to pipeline
        logger=logger
    )
    
    
    # Show what's in the config
    print(f"Config loaded from: {pipeline.config_manager.config_path}")
    print(f"Available experiments: {list(pipeline.config.get('experiment_configs', {}).keys())}")
    print(f"Available experiments: {list(pipeline.config_manager.config.get('experiment_configs', {}).keys())}")
    print(f"Available subjects: {list(pipeline.config.get('subjects', {}).keys())}")
    

    # Get subject list from pipeline
    # This will use the subject_selection logic from PipelineManager
    subject_list = pipeline.get_subject_list()
    
    print(f"Pipeline initialized with run ID: {pipeline.run_id}")
    print(f"Subject selection: {subject_selection}")
    print(f"Experiment config: {experiment_config}")
    print(f"Final subject list: {subject_list}")
    
    # Run session extraction if not skipped
    if not skip_extraction:
        logger.info("Main: Starting session extraction pipeline step...")
        try:
            pipeline.initialize_session_extractor(force=force_extraction)
            pipeline.extract_sessions(force=force_extraction)
            logger.info("Main: Session extraction pipeline step completed")
        except Exception as e:
            logger.error(f"Main: Session extraction pipeline step failed: {str(e)}")
            return False
    else:
        logger.info("Main: Skipping session extraction")
    
    # Run session preprocessing if not skipped
    if not skip_preprocessing:
        logger.info("Main: Starting session preprocessing pipeline step...")
        try:
            pipeline.initialize_session_preprocessor(force=force_preprocessing)
            pipeline.preprocess_sessions(force=force_preprocessing)
            logger.info("Main: Session preprocessing pipeline step completed")
        except Exception as e:
            logger.error(f"Main: Session preprocessing pipeline step failed: {str(e)}")
            return False
    else:
        logger.info("Main: Skipping session preprocessing")
    
    # Load preprocessed data if not skipped
    if not skip_loading:
        logger.info("Main: Starting data loading pipeline step...")
        try:
            pipeline.initialize_data_loader()
            loaded_data = pipeline.load_data()
            logger.info("Main: Data loading pipeline step completed")
            
            # Log summary of loaded data
            metadata = loaded_data['metadata']
            logger.info(f"Main: Loaded {metadata['total_sessions_loaded']}/{metadata['total_sessions_requested']} sessions from {metadata['subjects_loaded']}/{metadata['subjects_requested']} subjects")
            
        except Exception as e:
            logger.error(f"Main: Data loading pipeline step failed: {str(e)}")
            return False
    else:
        logger.info("Main: Skipping data loading")
        loaded_data = None
    
    # Run data analysis if not skipped
    if not skip_analysis and loaded_data is not None:
        logger.info("Main: Starting data analysis pipeline step...")
        try:
            pipeline.initialize_data_analyzer(loaded_data)
            analysis_results = pipeline.analyze_data(loaded_data)
            logger.info("Main: Data analysis pipeline step completed")
            
            # Log summary of analysis results
            logger.info(f"Main: Analysis completed for {analysis_results.get('subjects_analyzed', 0)} subjects")
            
        except Exception as e:
            logger.error(f"Main: Data analysis pipeline step failed: {str(e)}")
            return False
    else:
        if loaded_data is None:
            logger.info("Main: Skipping analysis - no data loaded")
        else:
            logger.info("Main: Skipping data analysis")
        analysis_results = None
    
    # Generate visualizations if not skipped
    if not skip_visualization and analysis_results is not None:
        logger.info("Main: Starting visualization pipeline step...")
        try:
            pipeline.initialize_visualizer(analysis_results)
            visualization_results = pipeline.generate_visualizations(analysis_results)
            logger.info("Main: Visualization pipeline step completed")
            
            # Log summary of visualization results
            logger.info(f"Main: Generated {visualization_results.get('figures_generated', 0)} figures")
            
        except Exception as e:
            logger.error(f"Main: Visualization pipeline step failed: {str(e)}")
            return False
    else:
        if analysis_results is None:
            logger.info("Main: Skipping visualization - no analysis results")
        else:
            logger.info("Main: Skipping visualization")
    
    # Generate reports if not skipped
    if not skip_reporting and analysis_results is not None:
        logger.info("Main: Starting report generation pipeline step...")
        try:
            pipeline.initialize_report_generator(analysis_results)
            report_results = pipeline.generate_reports(analysis_results)
            logger.info("Main: Report generation pipeline step completed")
            
            # Log summary of report results
            reports_generated = report_results.get('reports_generated', 0)
            logger.info(f"Main: Generated {reports_generated} reports")
            
        except Exception as e:
            logger.error(f"Main: Report generation pipeline step failed: {str(e)}")
    else:
        if analysis_results is None:
            logger.info("Main: Skipping report generation - no analysis results")
        else:
            logger.info("Main: Skipping report generation")
    
    # === IMAGING PREPROCESSING PIPELINE ===
    # Run imaging preprocessing after behavioral pipeline completes
    logger.info("Main: Checking for imaging preprocessing requirements...")
    
    # Configuration for imaging preprocessing
    imaging_experiment_config = "single_interval_discrimination"
    imaging_subject_selection = "YH24LG"  # Comma-separated list
    
    
    skip_imaging = True
    
    
    if not skip_imaging:
        # Initialize imaging pipeline
        try:
            imaging_pipeline = PipelineManager(
                config_path='modules/config/config.yaml',
                experiment_config=imaging_experiment_config,
                subject_selection=imaging_subject_selection,
                run_id=run_id,
                logger=logger
            )
            
            # Initialize Suite2p preprocessor
            imaging_pipeline.initialize_imaging_suite2p_preprocessor()
            logger.info("Main: Suite2p preprocessor initialized successfully")        
            
            # Run Suite2p preprocessing
            imaging_pipeline.preprocess_imaging_suite2p_sessions(force=False)
            logger.info("Main: Suite2p preprocessing completed successfully")
            # Initialize and run imaging analyzer
            logger.info("Main: Starting imaging analysis pipeline step...")
            
            # # Initialize imaging analyzer
            # imaging_pipeline.initialize_imaging_analyzer()
            # logger.info("Main: Imaging analyzer initialized successfully")
            
            # # Run imaging analysis
            # imaging_analysis_results = imaging_pipeline.analyze_imaging_sessions(force=True)
            # logger.info("Main: Imaging analysis pipeline step completed")
            
        except Exception as e:
            logger.error(f"Main: Failed to run imaging pipeline: {str(e)}")               
            
    # Configuration for imaging preprocessing
    imaging_experiment_config = "eye_blink_conditioning"
    imaging_subject_selection = "Test"  # Comma-separated list

    skip_ebc_imaging = False
        
    # if not skip_ebc_imaging:
    #     # Initialize imaging pipeline
    #     try:
    #         # imaging_pipeline = PipelineManager(
    #         #     config_path='modules/config/config.yaml',
    #         #     experiment_config=imaging_experiment_config,
    #         #     subject_selection=imaging_subject_selection,
    #         #     run_id=run_id,
    #         #     logger=logger
    #         # )
            

    #         # subjects = loaded_data['subjects']
    #         # for subject in subjects:
    #         #     print(subject)
    #         #     sessions = subjects[subject]['sessions']
    #         #     for session_id in sessions.keys():
    #         #         print(session_id)
    #         #         session = sessions[session_id]
                    
                    
            


    #         # # Initialize Suite2p preprocessor
    #         # imaging_pipeline.initialize_imaging_suite2p_preprocessor()
    #         # logger.info("Main: Suite2p preprocessor initialized successfully")        
            
    #         # # Run Suite2p preprocessing
    #         # imaging_pipeline.preprocess_imaging_suite2p_sessions(force=False)
    #         # logger.info("Main: Suite2p preprocessing completed successfully")
            
    #         # # Initialize and run imaging analyzer
    #         # logger.info("Main: Starting imaging analysis pipeline step...")            
    #         # # # Initialize imaging analyzer
    #         # imaging_pipeline.initialize_imaging_analyzer()
    #         # logger.info("Main: Imaging analyzer initialized successfully")
            
    #         # # # Run imaging analysis
    #         # imaging_analysis_results = imaging_pipeline.analyze_imaging_sessions(force=True)
    #         # logger.info("Main: Imaging analysis pipeline step completed")
               
    #     except Exception as e:
    #         logger.error(f"Main: Failed to run imaging pipeline: {str(e)}")   
 
        

    # parser = argparse.ArgumentParser(description='Experiments can go shit but Yicong will love you forever!')
    # parser = argparse.ArgumentParser(description='Experiments are going well...Tim')
    # parser.add_argument('--data_path',        required=True, type=str, help='Path to the 2P imaging data.')
    # parser.add_argument('--save_path',        required=True, type=str, help='Path to save the results.')
    from modules.utils import run_suite2p_pipeline as s2p
    from types import SimpleNamespace
    args = SimpleNamespace()
    
    args.denoise = 0
    args.spatial_scale = 1
    args.data_path = 'C://behavior//session_data//Test'
    args.save_path = 'C://behavior//session_data//Test'
    args.nchannels = 2
    args.functional_chan = 2
    args.target_structure = 'neuron'
    
    ops, db = s2p.set_params(args)
    s2p.process_vol(args)
 
    import os
    import h5py
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        subjects = loaded_data['subjects']
        for subject in subjects:
            print(subject)
            sessions = subjects[subject]['sessions']
            for session_id in sessions.keys():
                print(session_id)
                session = sessions[session_id]
                df_trials = session['df_trials']
         
                voltage_data = {}
                            
                voltage_file = os.path.join(args.data_path, 'raw_voltages.h5')

                # get voltage data              
                with h5py.File(voltage_file, 'r') as f:                    
                    raw_group = f['raw']
                    
                    # vol recording reports time in ms, convert to s
                    ms_per_s = 1000
                    voltage_data['vol_time'] = raw_group['vol_time'][:] / ms_per_s
                    # calculate duration and sampling rate
                    voltage_data['vol_n_samples'] = len(voltage_data['vol_time'])
                    voltage_data['vol_duration'] = voltage_data['vol_time'][-1]                
                    voltage_data['voltage_fs'] = voltage_data['vol_n_samples'] / voltage_data['vol_duration']
        
                    # Load voltage channels directly to named variables
                    # Load directly to RAM
                    voltage_data['vol_start'] = raw_group['vol_start'][:] if 'vol_start' in raw_group else None
                    voltage_data['vol_stim_vis'] = raw_group['vol_stim_vis'][:] if 'vol_stim_vis' in raw_group else None
                    # voltage_data['vol_hifi'] = raw_group['vol_hifi'][:] if 'vol_hifi' in raw_group else None
                    voltage_data['vol_img'] = raw_group['vol_img'][:] if 'vol_img' in raw_group else None
                    voltage_data['vol_stim_aud'] = raw_group['vol_stim_aud'][:] if 'vol_stim_aud' in raw_group else None
                    # voltage_data['vol_flir'] = raw_group['vol_flir'][:] if 'vol_flir' in raw_group else None
                    # voltage_data['vol_pmt'] = raw_group['vol_pmt'][:] if 'vol_pmt' in raw_group else None
                    # voltage_data['vol_led'] = raw_group['vol_led'][:] if 'vol_led' in raw_group else None
   
                # Plot vol_start over time for verification
                plt.figure(figsize=(8, 4))
                plt.plot(voltage_data['vol_time'], voltage_data['vol_start'])
                plt.xlabel("Time (s)")
                plt.ylabel("Start voltage")
                plt.title("vol_start vs vol_time")
                plt.tight_layout()
                plt.show()   


                # get vol_start and vol_time
                vol_start = np.asarray(voltage_data['vol_start'])
                vol_time = np.asarray(voltage_data['vol_time'])

                rising_edge_idx = np.where((vol_start[:-1] == 0) & (vol_start[1:] == 1))[0] + 1
                falling_edge_idx = np.where((vol_start[:-1] == 1) & (vol_start[1:] == 0))[0] + 1

                rise_times = vol_time[rising_edge_idx]
                fall_times = vol_time[falling_edge_idx]

                voltage_data['vol_time_aligned'] = vol_time - rise_times[0]

                voltage_data['rise_times'] = rise_times
                voltage_data['fall_times'] = fall_times

                block_gap_threshold = 1.0
                if rise_times.size:
                    block_breaks = np.where(np.diff(rise_times) > block_gap_threshold)[0] + 1
                    block_indices = np.split(np.arange(len(rise_times)), block_breaks)
                    trial_cam_start = [rise_times[idx_block[0]] for idx_block in block_indices if len(idx_block) > 0]
                    trial_cam_stop = [rise_times[idx_block[-1]] for idx_block in block_indices if len(idx_block) > 0]
                else:
                    trial_cam_start, trial_cam_stop = [], []

                df_trials['trial_cam_start'] = pd.Series(trial_cam_start, index=df_trials.index)
                df_trials['trial_cam_stop'] = pd.Series(trial_cam_stop, index=df_trials.index)


                for trial_idx, row in df_trials.iterrows():
                    trial_start = row['trial_start_timestamp']
                    trial_stop = row['trial_stop_timestamp']

                    times = np.asarray(row['Times'])
                    positions = np.asarray(row['Positions'])
                    positions_unwrapped = np.asarray(row['PositionsUnwrapped'])
                    linear_positions = np.asarray(row['LinearPositions'])
                    fec_times = np.asarray(row['FECTimes'])
                    fec_values = np.asarray(row['FEC'])
                    vol_time_arr = np.asarray(voltage_data.get('vol_time_aligned', []))
                    vol_start_arr = np.asarray(voltage_data.get('vol_start', []))

                    if times.size == 0 or fec_times.size == 0 or vol_time_arr.size == 0:
                        continue

                    time_mask = (times >= 0) & (times <= (trial_stop-trial_start))
                    fec_mask = (fec_times >= 0) & (fec_times <= (trial_stop-trial_start))
                    vol_mask = (vol_time_arr >= trial_start) & (vol_time_arr <= trial_stop)

                    rel_times = times[time_mask]
                    rel_fec_times = fec_times[fec_mask]
                    rel_vol_times = vol_time_arr[vol_mask] - trial_start

                    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(8, 12))
                    axes[0].plot(rel_times, positions[time_mask], color='tab:blue')
                    axes[0].set_ylabel("Positions (deg)")

                    axes[1].plot(rel_times, positions_unwrapped[time_mask], color='tab:orange')
                    axes[1].set_ylabel("PositionsUnwrapped (deg)")

                    axes[2].plot(rel_times, linear_positions[time_mask], color='tab:green')
                    axes[2].set_ylabel("LinearPositions (m)")

                    axes[3].plot(rel_fec_times, fec_values[fec_mask], color='tab:red')
                    axes[3].set_ylabel("FEC (%)")

                    axes[4].step(rel_vol_times, vol_start_arr[vol_mask], where='post', color='tab:purple')
                    axes[4].set_ylabel("vol_start")
                    axes[4].set_xlabel("Time since trial start (s)")

                    for ax in axes:
                        ax.axvline(trial_stop - trial_start, color='k', linestyle='--', alpha=0.4)
                        ax.grid(True, alpha=0.3)

                    fig.suptitle(f"Trial {trial_idx}")
                    plt.tight_layout()
                    plt.show()







    except Exception as e:
        logger.error(f"Main: Failed to run imaging pipeline: {str(e)}")         
         
            
         
       

    

    
 
        
# %%        
        
    # path = 'D:/behavior/2p_imaging/processed/2afc/YH24LG/YH24LG_CRBL_lobulev_20250620_2afc-494/sid_imaging_segmented_data.pkl'

    # import pickle

    # with open(path, 'rb') as f:
    #     trial_data = pickle.load(f)   # one object back (e.g., a dict)  
     
        
# %%          
    
   
    print('')
        
        # # Log summary of imaging analysis results
        # if imaging_analysis_results.get('success', False):
        #     subjects_processed = imaging_analysis_results.get('subjects_processed', 0)
        #     subjects_failed = imaging_analysis_results.get('subjects_failed', 0)
        #     logger.info(f"Main: Imaging analysis completed for {subjects_processed} subjects, {subjects_failed} failed")
        # else:
        #     logger.error(f"Main: Imaging analysis failed: {imaging_analysis_results.get('error', 'Unknown error')}")
        

# %%   
    # For now, just return success
    # return True

def run_command_line_mode(run_id, logger):
    """Minimal command line mode to prevent crashes"""
    logger.info("Main: === Data Analysis Pipeline - Command Line Mode ===")
    return True

def cleanup_logger(logger=None):
    """Clean up logger handlers to prevent accumulation in Spyder."""
    # If no specific logger passed, clean up all loggers
    if logger is not None:
        # Close and remove handlers from specific logger
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
    
    # Always clean up root logger and all named loggers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)
    
    # Clean up all named loggers that might have been created
    for logger_name in list(logging.root.manager.loggerDict.keys()):
        named_logger = logging.getLogger(logger_name)
        for handler in named_logger.handlers[:]:
            handler.close()
            named_logger.removeHandler(handler)
        named_logger.handlers.clear()
        named_logger.setLevel(logging.NOTSET)

if __name__ == "__main__":
    try:
        success = main()
        print(f"Main completed with success: {success}")
    finally:
        # Clean up all logger handlers to prevent accumulation in Spyder
        print('Cleaning up loggers...')
        cleanup_logger()
        print('Logger cleanup completed')
    # Temporarily remove sys.exit to see full output
    # sys.exit(0 if success else 1)