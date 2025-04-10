from utils import directories

subject_list = {
    'list_session_name' : {
        
    },
    'session_folder' : '',
    # 'sig_tag' : '',
    # 'force_label' : None,
}

# Define config
paths = {
    'output_dir_onedrive': directories.OUTPUT_DIR_ONEDRIVE,
    'output_dir_local': directories.OUTPUT_DIR_LOCAL,
    'figure_dir_local': directories.FIGURE_DIR_LOCAL,
    'registry_path': directories.FIGURE_REGISTRY,
}

session_config_TS01 = {
    'list_session_name' : {
        'LCHR_TS01_single_interval_discrimination_V_1_10_20250406_165117',
        'LCHR_TS01_single_interval_discrimination_V_1_10_20250405_113729',
        # 'LCHR_TS01_single_interval_discrimination_V_1_10_20250404_231556',
        'LCHR_TS01_single_interval_discrimination_V_1_10_20250403_202240',
        },
    'subject_name' : 'TS01',
    'session_folder' : 'LCHR_TS01_update',
    'sig_tag' : 'all',
    'force_label' : None,
    }      

session_config_TS02 = {
    'list_session_name' : {
        'LCHR_TS02_single_interval_discrimination_V_1_10_20250316_220701',
        'LCHR_TS02_single_interval_discrimination_V_1_10_20250317_223308',
        'LCHR_TS02_single_interval_discrimination_V_1_10_20250331_150124',
        # 'LCHR_TS02_single_interval_discrimination_V_1_10_20250401_163804',
        'LCHR_TS02_single_interval_discrimination_V_1_10_20250403_210843',
        'LCHR_TS02_single_interval_discrimination_V_1_10_20250404_002251',
        },
    'subject_name' : 'TS02',
    'session_folder' : 'LCHR_TS02_update',
    'sig_tag' : 'all',
    'force_label' : None,
    }     

session_config_list_TS02 = {
    'list_config': [
        session_config_TS02,
        ],
    'label_names' : {
        '-1':'Exc',
        '1':'Inh_VIP',
        '2':'Inh_SST',
        },
    'subject_name' : 'TS02',
    'output_filename' : 'TS03_CRBL_passive.html'
    }     

session_config_list_2AFC = {
    'list_config': [
        session_config_TS01,
        session_config_TS02,
        ],
    'label_names' : {
        '-1':'Exc',
        '1':'Inh_VIP',
        '2':'Inh_SST',
        },
    'subject_name' : 'all',
    'output_filename' : 'all_PPC_passive.html',  
    'paths' : paths,
    } 


