import modules.config.directories as directories

subject_list = {
    'list_session_name' : {
        
    },
    'session_folder' : '',
    # 'sig_tag' : '',
    # 'force_label' : None,
}

# Define config
paths = {
    'session_data' : directories.SESSION_DATA_PATH,
    'extracted_data' : directories.EXTRACTED_DATA_PATH,
    'preprocessed_data' : directories.PREPROCESSED_DATA_PATH,
    'output_dir_onedrive': directories.OUTPUT_DIR_ONEDRIVE,
    'output_dir_local': directories.OUTPUT_DIR_LOCAL,
    'figure_dir_local': directories.FIGURE_DIR_LOCAL,
    'registry_path': directories.FIGURE_REGISTRY,
    'error_log_path': directories.ERROR_LOG,
}


# pdf_pg_cover = {
#     "fig_size": (30, 15),     # final combined PDF size in inches (width, height)
#     "grid_size": (4, 8),    # (nrows, ncols) for Gridspec layout
#     "dpi": 300,               # DPI for rasterized image placement
#     "margins": {              # optional, if you want to customize margins
#         "left": 0,
#         "right": 0,
#         "top": 0,
#         "bottom": 0,
#         "wspace": 0,
#         "hspace": 0,
#     },
# }


# pdf_spec = {
#     "pdf_pg_cover": pdf_pg_cover,
#     "pdf_pg_opto_psychometric": pdf_pg_opto_psychometric,
#     "pdf_pg_opto_psychometric_residual": pdf_pg_opto_psychometric_residual,
#     "pdf_pg_licking" : pdf_pg_licking,
#     "pdf_pg_avg_licking" : pdf_pg_avg_licking,
# }


session_config_TS01 = {
    'list_session_name' : [
        'LCHR_TS01_single_interval_discrimination_V_1_10_20250417_075031',
        'LCHR_TS01_single_interval_discrimination_V_1_10_20250416_225247',
        'LCHR_TS01_single_interval_discrimination_V_1_10_20250414_073145',
        'LCHR_TS01_single_interval_discrimination_V_1_10_20250413_111016',
        'LCHR_TS01_single_interval_discrimination_V_1_10_20250411_112942',
        'LCHR_TS01_single_interval_discrimination_V_1_10_20250410_125310',
        'LCHR_TS01_single_interval_discrimination_V_1_10_20250409_190250',
        'LCHR_TS01_single_interval_discrimination_V_1_10_20250407_100841',    # right bias, could be residual
        
        'LCHR_TS01_single_interval_discrimination_V_1_10_20250406_165117',   # right bias, could be residual
        'LCHR_TS01_single_interval_discrimination_V_1_10_20250405_113729',     # right bias, could be residual
        
        'LCHR_TS01_single_interval_discrimination_V_1_10_20250404_231556',
        
        'LCHR_TS01_single_interval_discrimination_V_1_10_20250403_202240',
        
        'LCHR_TS01_single_interval_discrimination_V_1_10_20250401_154509',
        'LCHR_TS01_single_interval_discrimination_V_1_10_20250331_141036',
        # 'LCHR_TS01_single_interval_discrimination_V_1_10_20250330_132843',    # flat control, could be residual
        
        # 'LCHR_TS01_single_interval_discrimination_V_1_8_20250329_113237',    # control
        # 'LCHR_TS01_single_interval_discrimination_V_1_8_20250328_110455',    # control
        
        # 'LCHR_TS01_single_interval_discrimination_V_1_10_20250327_184557',    # flat control, could be residual
        # 'LCHR_TS01_single_interval_discrimination_V_1_10_20250326_204535'    # flat control, could be residual
        'LCHR_TS01_single_interval_discrimination_V_1_10_20250324_202107',
        # 'LCHR_TS01_single_interval_discrimination_V_1_10_20250321_195823',   # flat control, could be residual
        # 'LCHR_TS01_single_interval_discrimination_V_1_10_20250320_203341',   # flat control, could be residual
        'LCHR_TS01_single_interval_discrimination_V_1_10_20250319_234330',
        'LCHR_TS01_single_interval_discrimination_V_1_10_20250318_184345',
        'LCHR_TS01_single_interval_discrimination_V_1_10_20250317_213448',
        'LCHR_TS01_single_interval_discrimination_V_1_10_20250316_213144',
        
        'LCHR_TS01_single_interval_discrimination_V_1_10_20250315_215016',     # mid1
        
        'LCHR_TS01_single_interval_discrimination_V_1_10_20250314_205555',    # low pwr
        'LCHR_TS01_single_interval_discrimination_V_1_10_20250312_215826',    # low pwr    
        'LCHR_TS01_single_interval_discrimination_V_1_10_20250311_195050',    # low pwr
        'LCHR_TS01_single_interval_discrimination_V_1_9_20250309_220849',    # low pwr
        'LCHR_TS01_single_interval_discrimination_V_1_9_20250308_214653',    # low pwr
        'LCHR_TS01_single_interval_discrimination_V_1_9_20250306_202610',    # low pwr
        'LCHR_TS01_single_interval_discrimination_V_1_9_20250305_163852',    # low pwr
        'LCHR_TS01_single_interval_discrimination_V_1_9_20250304_195615',    # low pwr
        'LCHR_TS01_single_interval_discrimination_V_1_9_20250303_205316',    # low pwr
        'LCHR_TS01_single_interval_discrimination_V_1_9_20250302_212005',    # low pwr
        'LCHR_TS01_single_interval_discrimination_V_1_9_20250301_213418',    # low pwr
        'LCHR_TS01_single_interval_discrimination_V_1_9_20250227_195132',    # low pwr
        'LCHR_TS01_single_interval_discrimination_V_1_9_20250226_204526',    # low pwr
        'LCHR_TS01_single_interval_discrimination_V_1_9_20250225_191740',    # low pwr
        'LCHR_TS01_single_interval_discrimination_V_1_9_20250224_221618',    # low pwr
        ],
    'subject_name' : 'TS01_LChR',
    'session_folder' : 'LCHR_MC01',
    }   

session_config_YH24 = {
    'list_session_name' : [
        # 'TS03_single_interval_discrimination_V_1_11_20250425_073442',   # 
        'YH24LG_single_interval_discrimination_V_1_11_20250425_193529',   #       
        ],
    'subject_name' : 'YH24_LG',
    'session_folder' : 'YH24',
    'sig_tag' : 'all',
    'force_label' : None,
    }

session_config_TS03 = {
    'list_session_name' : [
        # 'TS03_single_interval_discrimination_V_1_11_20250425_073442',   # 
        'TS03_single_interval_discrimination_V_1_11_20250424_071549',   #       
        ],
    'subject_name' : 'TS03_LG',
    'session_folder' : 'TS03',
    'sig_tag' : 'all',
    'force_label' : None,
    } 

# session_config_TS01 = {
#     'list_session_name' : [
#         'LCHR_TS01_single_interval_discrimination_V_1_10_20250417_075031',
#         'LCHR_TS01_single_interval_discrimination_V_1_10_20250416_225247',
#         'LCHR_TS01_single_interval_discrimination_V_1_10_20250414_073145',
#         'LCHR_TS01_single_interval_discrimination_V_1_10_20250413_111016',
#         'LCHR_TS01_single_interval_discrimination_V_1_10_20250411_112942',
#         'LCHR_TS01_single_interval_discrimination_V_1_10_20250410_125310',
#         'LCHR_TS01_single_interval_discrimination_V_1_10_20250409_190250',
#         'LCHR_TS01_single_interval_discrimination_V_1_10_20250407_100841',    # right bias, could be residual
        
#         'LCHR_TS01_single_interval_discrimination_V_1_10_20250406_165117',   # right bias, could be residual
#         'LCHR_TS01_single_interval_discrimination_V_1_10_20250405_113729',     # right bias, could be residual
        
#         'LCHR_TS01_single_interval_discrimination_V_1_10_20250404_231556',
        
#         'LCHR_TS01_single_interval_discrimination_V_1_10_20250403_202240',
        
#         'LCHR_TS01_single_interval_discrimination_V_1_10_20250401_154509',
#         'LCHR_TS01_single_interval_discrimination_V_1_10_20250331_141036',
#         # 'LCHR_TS01_single_interval_discrimination_V_1_10_20250330_132843',    # flat control, could be residual
        
#         # 'LCHR_TS01_single_interval_discrimination_V_1_8_20250329_113237',    # control
#         # 'LCHR_TS01_single_interval_discrimination_V_1_8_20250328_110455',    # control
        
#         # 'LCHR_TS01_single_interval_discrimination_V_1_10_20250327_184557',    # flat control, could be residual
#         # 'LCHR_TS01_single_interval_discrimination_V_1_10_20250326_204535'    # flat control, could be residual
#         'LCHR_TS01_single_interval_discrimination_V_1_10_20250324_202107',
#         # 'LCHR_TS01_single_interval_discrimination_V_1_10_20250321_195823',   # flat control, could be residual
#         # 'LCHR_TS01_single_interval_discrimination_V_1_10_20250320_203341',   # flat control, could be residual
#         'LCHR_TS01_single_interval_discrimination_V_1_10_20250319_234330',
#         'LCHR_TS01_single_interval_discrimination_V_1_10_20250318_184345',
#         'LCHR_TS01_single_interval_discrimination_V_1_10_20250317_213448',
#         'LCHR_TS01_single_interval_discrimination_V_1_10_20250316_213144',
        
#         'LCHR_TS01_single_interval_discrimination_V_1_10_20250315_215016',     # mid1
        
#         'LCHR_TS01_single_interval_discrimination_V_1_10_20250314_205555',    # low pwr
#         'LCHR_TS01_single_interval_discrimination_V_1_10_20250312_215826',    # low pwr    
#         'LCHR_TS01_single_interval_discrimination_V_1_10_20250311_195050',    # low pwr
#         'LCHR_TS01_single_interval_discrimination_V_1_9_20250309_220849',    # low pwr
#         'LCHR_TS01_single_interval_discrimination_V_1_9_20250308_214653',    # low pwr
#         'LCHR_TS01_single_interval_discrimination_V_1_9_20250306_202610',    # low pwr
#         'LCHR_TS01_single_interval_discrimination_V_1_9_20250305_163852',    # low pwr
#         'LCHR_TS01_single_interval_discrimination_V_1_9_20250304_195615',    # low pwr
#         'LCHR_TS01_single_interval_discrimination_V_1_9_20250303_205316',    # low pwr
#         'LCHR_TS01_single_interval_discrimination_V_1_9_20250302_212005',    # low pwr
#         'LCHR_TS01_single_interval_discrimination_V_1_9_20250301_213418',    # low pwr
#         'LCHR_TS01_single_interval_discrimination_V_1_9_20250227_195132',    # low pwr
#         'LCHR_TS01_single_interval_discrimination_V_1_9_20250226_204526',    # low pwr
#         'LCHR_TS01_single_interval_discrimination_V_1_9_20250225_191740',    # low pwr
#         'LCHR_TS01_single_interval_discrimination_V_1_9_20250224_221618',    # low pwr
#         ],
#     'subject_name' : 'TS01_LChR',
#     'session_folder' : 'LCHR_TS01',
#     'sig_tag' : 'all',
#     'force_label' : None,
#     }   

# session_config_TS01 = {
#     'list_session_name' : [
#         'LCHR_TS01_single_interval_discrimination_V_1_10_20250406_165117',
#         'LCHR_TS01_single_interval_discrimination_V_1_10_20250405_113729',
#         # 'LCHR_TS01_single_interval_discrimination_V_1_10_20250404_231556',
#         'LCHR_TS01_single_interval_discrimination_V_1_10_20250403_202240',
        
#         'LCHR_TS01_single_interval_discrimination_V_1_10_20250401_154509',
#         'LCHR_TS01_single_interval_discrimination_V_1_10_20250331_141036',
#         'LCHR_TS01_single_interval_discrimination_V_1_10_20250330_132843',
#         # 'LCHR_TS01_single_interval_discrimination_V_1_8_20250329_113237',
#         # 'LCHR_TS01_single_interval_discrimination_V_1_8_20250328_110455',
#         'LCHR_TS01_single_interval_discrimination_V_1_10_20250327_184557',
#         'LCHR_TS01_single_interval_discrimination_V_1_10_20250326_204535',
#         'LCHR_TS01_single_interval_discrimination_V_1_10_20250324_202107',
#         'LCHR_TS01_single_interval_discrimination_V_1_10_20250321_195823',
#         'LCHR_TS01_single_interval_discrimination_V_1_10_20250320_203341',
#         'LCHR_TS01_single_interval_discrimination_V_1_10_20250319_234330',
#         'LCHR_TS01_single_interval_discrimination_V_1_10_20250318_184345',
#         'LCHR_TS01_single_interval_discrimination_V_1_10_20250317_213448',
#         'LCHR_TS01_single_interval_discrimination_V_1_10_20250316_213144',
#         # 'LCHR_TS01_single_interval_discrimination_V_1_10_20250315_215016',
#         ],
#     'subject_name' : 'TS01',
#     'session_folder' : 'LCHR_TS01',
#     'sig_tag' : 'all',
#     'force_label' : None,
#     }      


# session_config_TS02 = {
#     'list_session_name' : [
#        'LCHR_TS02_single_interval_discrimination_V_1_10_20250417_085055',    # 500/2000 isi
#        # 'LCHR_TS02_single_interval_discrimination_V_1_10_20250416_220127',    # 500/2000 isi
#         ],
#     'subject_name' : 'TS02_LChR',
#     'session_folder' : 'LCHR_TS02',
#     'sig_tag' : 'all',
#     'force_label' : None,
#     }          

# session_config_TS02 = {
#     'list_session_name' : [
#         'LCHR_TS02_single_interval_discrimination_V_1_10_20250317_223308',       
#         ],
#     'subject_name' : 'TS02',
#     'session_folder' : 'LCHR_TS02',
#     'sig_tag' : 'all',
#     'force_label' : None,
#     }   

session_config_TS02 = {
    'list_session_name' : [
       # 'LCHR_TS02_single_interval_discrimination_V_1_10_20250417_085055',    # 500/2000 isi
       # 'LCHR_TS02_single_interval_discrimination_V_1_10_20250416_220127',    # 500/2000 isi
       # 'LCHR_TS02_single_interval_discrimination_V_1_10_20250415_145638',    # 500/2000 isi
       # 'LCHR_TS02_single_interval_discrimination_V_1_10_20250414_172542',
        
        
       # 'LCHR_TS02_single_interval_discrimination_V_1_10_20250411_103435',    # 500/2000 isi
       'LCHR_TS02_single_interval_discrimination_V_1_10_20250410_074411',
       'LCHR_TS02_single_interval_discrimination_V_1_10_20250409_201614',      
       'LCHR_TS02_single_interval_discrimination_V_1_10_20250407_191414',
       # # # 'LCHR_TS02_single_interval_discrimination_V_1_10_20250406_173626',    # exclude
       'LCHR_TS02_single_interval_discrimination_V_1_10_20250405_124545',
           
        
        'LCHR_TS02_single_interval_discrimination_V_1_10_20250404_002251',        
        'LCHR_TS02_single_interval_discrimination_V_1_10_20250403_210843',
       # #  # 'LCHR_TS02_single_interval_discrimination_V_1_10_20250401_163804',   # exclude     

       #  'LCHR_TS02_single_interval_discrimination_V_1_10_20250331_150124',
       # #  # 'LCHR_TS02_single_interval_discrimination_V_1_10_20250327_174456',   # exclude
       # #  # 'LCHR_TS02_single_interval_discrimination_V_1_10_20250326_134254',   # exclude   
       # #  # 'LCHR_TS02_single_interval_discrimination_V_1_10_20250324_211801',   # exclude
       # #  # 'LCHR_TS02_single_interval_discrimination_V_1_10_20250321_204016',   # exclude
        'LCHR_TS02_single_interval_discrimination_V_1_10_20250320_213030',
       # #  # 'LCHR_TS02_single_interval_discrimination_V_1_10_20250319_224052',   # exclude
        'LCHR_TS02_single_interval_discrimination_V_1_10_20250318_175930',           
        

        'LCHR_TS02_single_interval_discrimination_V_1_10_20250317_223308',
        'LCHR_TS02_single_interval_discrimination_V_1_10_20250316_220701',
        'LCHR_TS02_single_interval_discrimination_V_1_10_20250315_225512',
       
       
       # # 'LCHR_TS02_single_interval_discrimination_V_1_10_20250314_220109',     # mid1  
       'LCHR_TS02_single_interval_discrimination_V_1_10_20250311_210500',       
       'LCHR_TS02_single_interval_discrimination_V_1_9_20250309_230705',       
       'LCHR_TS02_single_interval_discrimination_V_1_9_20250308_230024',       
       'LCHR_TS02_single_interval_discrimination_V_1_9_20250306_212416',       
       'LCHR_TS02_single_interval_discrimination_V_1_9_20250305_175626',       
       'LCHR_TS02_single_interval_discrimination_V_1_9_20250304_205131',       
       'LCHR_TS02_single_interval_discrimination_V_1_9_20250303_215451',       
       'LCHR_TS02_single_interval_discrimination_V_1_9_20250302_222506',       
       'LCHR_TS02_single_interval_discrimination_V_1_9_20250301_223020',       
       'LCHR_TS02_single_interval_discrimination_V_1_9_20250227_205609',       
       'LCHR_TS02_single_interval_discrimination_V_1_9_20250225_201421',       
       'LCHR_TS02_single_interval_discrimination_V_1_9_20250224_231949',       
        ],
    'subject_name' : 'TS02',
    'session_folder' : 'LCHR_MC02',
    'sig_tag' : 'all',
    'force_label' : None,
    }     

session_config_TS06 = {
    'list_session_name' : [
        'SCHR_TS06_single_interval_discrimination_V_1_10_20250416_205942',        
        'SCHR_TS06_single_interval_discrimination_V_1_10_20250415_091444',        
        'SCHR_TS06_single_interval_discrimination_V_1_10_20250414_093242',    # 400-1100 isi       
        
        
        'SCHR_TS06_single_interval_discrimination_V_1_10_20250411_092808',        
        'SCHR_TS06_single_interval_discrimination_V_1_10_20250410_181115',
        'SCHR_TS06_single_interval_discrimination_V_1_10_20250409_091345',
        'SCHR_TS06_single_interval_discrimination_V_1_10_20250408_081805',
        'SCHR_TS06_single_interval_discrimination_V_1_10_20250407_064206',
        'SCHR_TS06_single_interval_discrimination_V_1_10_20250406_150431',
        'SCHR_TS06_single_interval_discrimination_V_1_10_20250405_170954',
        'SCHR_TS06_single_interval_discrimination_V_1_10_20250404_215615',
        'SCHR_TS06_single_interval_discrimination_V_1_10_20250403_080400',    # 400-1100 isi
        'SCHR_TS06_single_interval_discrimination_V_1_10_20250401_121236',    # 400-1100 isi
        'SCHR_TS06_single_interval_discrimination_V_1_10_20250330_113915',    # 400-1100 isi
        
        'SCHR_TS06_single_interval_discrimination_V_1_10_20250329_122935',   # r posterior
        ],
    'subject_name' : 'TS06',
    'session_folder' : 'SCHR_MC06',
    'sig_tag' : 'all',
    'force_label' : None,
    }   

session_config_TS07 = {
    'list_session_name' : [
        # 'SCHR_TS07_single_interval_discrimination_V_1_10_20250416_200308',  #  control, 500-2000 isi
        # 'SCHR_TS07_single_interval_discrimination_V_1_10_20250415_133410',  #  control, 500-2000 isi
        # 'SCHR_TS07_single_interval_discrimination_V_1_10_20250414_111734',  #  control, 500-2000 isi

        # 'SCHR_TS07_single_interval_discrimination_V_1_10_20250413_101621',  #  control, 500-2000 isi
        # 'SCHR_TS07_single_interval_discrimination_V_1_10_20250411_125506',  #  control, 500-2000 isi
        # 'SCHR_TS07_single_interval_discrimination_V_1_10_20250410_191332',  #  control, 500-2000 isi
        # 'SCHR_TS07_single_interval_discrimination_V_1_10_20250409_100600',  #  control, 500-2000 isi
        
        'SCHR_TS07_single_interval_discrimination_V_1_10_20250408_091223',   # 500/2000 isi, l posterior
        'SCHR_TS07_single_interval_discrimination_V_1_10_20250407_073549',   # 500/2000 isi, l posterior
        # 'SCHR_TS07_single_interval_discrimination_V_1_10_20250406_155535',   # flat control,  l posterior, 500/2000 isi
        
        # 'SCHR_TS07_single_interval_discrimination_V_1_10_20250405_183708',   # flat control, r posterior,500/2000 isi
        'SCHR_TS07_single_interval_discrimination_V_1_10_20250330_123319',   #  r posterior,500/2000 isi
        'SCHR_TS07_single_interval_discrimination_V_1_10_20250329_140842',   #  r posterior,500/2000 isi
        'SCHR_TS07_single_interval_discrimination_V_1_10_20250328_130500',   #  r posterior,500/2000 isi
        
        # 'SCHR_TS07_single_interval_discrimination_V_1_10_20250327_200019',    # mpfc, 500/2000 isi
        
        'SCHR_TS07_single_interval_discrimination_V_1_10_20250326_181348',   # r posterior, 500/2000 isi
        ],
    'subject_name' : 'TS07',
    'session_folder' : 'SCHR_MC07',
    'sig_tag' : 'all',
    'force_label' : None,
    }  

session_config_TS08 = {
    'list_session_name' : [
        'SCHR_TS08_single_interval_discrimination_V_1_10_20250416_181214',   #  500-2000 isi
        'SCHR_TS08_single_interval_discrimination_V_1_10_20250415_072926',   #  500-2000 isi
        # 'SCHR_TS08_single_interval_discrimination_V_1_10_20250414_144258',   #  right bias, 500-2000 isi
        
        
        
        'SCHR_TS08_single_interval_discrimination_V_1_10_20250411_080113',   #  500-2000 isi
        'SCHR_TS08_single_interval_discrimination_V_1_10_20250410_083251',   #  500-2000 isi
        # 'SCHR_TS08_single_interval_discrimination_V_1_10_20250409_072225',   #  right bias, 500-2000 isi
        'SCHR_TS08_single_interval_discrimination_V_1_10_20250408_101149',   #  500-2000 isi
        'SCHR_TS08_single_interval_discrimination_V_1_10_20250407_081958',   #  500-2000 isi        
        'SCHR_TS08_single_interval_discrimination_V_1_10_20250406_125720',   #  500-2000 isi
        'SCHR_TS08_single_interval_discrimination_V_1_10_20250405_193322',   #  500-2000 isi
        'SCHR_TS08_single_interval_discrimination_V_1_10_20250404_190523',   #  500-2000 isi
        
        # # 'SCHR_TS08_single_interval_discrimination_V_1_10_20250401_133940',   #  right bias, 400-1100 isi
        # 'SCHR_TS08_single_interval_discrimination_V_1_10_20250331_112110',   #  400-1100 isi
        # # 'SCHR_TS08_single_interval_discrimination_V_1_10_20250330_095530',   #  flat control,400-1100 isi
        # 'SCHR_TS08_single_interval_discrimination_V_1_10_20250329_103723',   #  400-1100 isi
        # # 'SCHR_TS08_single_interval_discrimination_V_1_10_20250328_140930',   #  flat control,400-1100 isi
        # 'SCHR_TS08_single_interval_discrimination_V_1_10_20250327_211331',   #  400-1100 isi
        ],
    'subject_name' : 'TS08',
    'session_folder' : 'SCHR_MC08',
    'sig_tag' : 'all',
    'force_label' : None,
    }  

session_config_TS09 = {
    'list_session_name' : [
        'SCHR_TS09_single_interval_discrimination_V_1_10_20250416_190712',   #  500-2000 isi
        'SCHR_TS09_single_interval_discrimination_V_1_10_20250415_081728',   #  500-2000 isi
        'SCHR_TS09_single_interval_discrimination_V_1_10_20250414_153403',   #  500-2000 isi
        
        
        'SCHR_TS09_single_interval_discrimination_V_1_10_20250411_083631',   #  500-2000 isi
        'SCHR_TS09_single_interval_discrimination_V_1_10_20250410_092709',   #  500-2000 isi
        'SCHR_TS09_single_interval_discrimination_V_1_10_20250409_081814',   #  500-2000 isi
        'SCHR_TS09_single_interval_discrimination_V_1_10_20250408_110157',   #  500-2000 isi
        'SCHR_TS09_single_interval_discrimination_V_1_10_20250407_091521',   #  500-2000 isi
        'SCHR_TS09_single_interval_discrimination_V_1_10_20250406_141128',   #  500-2000 isi
        'SCHR_TS09_single_interval_discrimination_V_1_10_20250405_202305',   #  500-2000 isi
        'SCHR_TS09_single_interval_discrimination_V_1_10_20250404_203608',   #  500-2000 isi
        'SCHR_TS09_single_interval_discrimination_V_1_10_20250403_192913',   #  500-2000 isi
        'SCHR_TS09_single_interval_discrimination_V_1_10_20250401_144409',   #  500-2000 isi
        'SCHR_TS09_single_interval_discrimination_V_1_10_20250331_121221',   #  500-2000 isi
        'SCHR_TS09_single_interval_discrimination_V_1_10_20250330_104559',   #  500-2000 isi
        'SCHR_TS09_single_interval_discrimination_V_1_10_20250329_112423',   #  500-2000 isi
        # 'SCHR_TS09_single_interval_discrimination_V_1_10_20250328_145956',   #  left bias,  500-2000 isi
        'SCHR_TS09_single_interval_discrimination_V_1_10_20250327_220834',   #  500-2000 isi        
        ],
    'subject_name' : 'TS09',
    'session_folder' : 'SCHR_MC09',
    'sig_tag' : 'all',
    'force_label' : None,
    }  


session_config_list_2AFC = {
    'list_config': {
        'LCHR_MC01': session_config_TS01,
        'LCHR_MC02': session_config_TS02,
        'SCHR_MC06': session_config_TS06,
        'SCHR_MC07': session_config_TS07,
        'SCHR_MC08': session_config_TS08,
        'SCHR_MC09': session_config_TS09,
        'TS03': session_config_TS03,
        'YH24': session_config_YH24,
    },
    'preprocessor': paths,
    'paths': paths,
}

# session_config_list_TS02 = {
#     'list_config': [
#         session_config_TS02,
#         ],
#     'label_names' : {
#         '-1':'Exc',
#         '1':'Inh_VIP',
#         '2':'Inh_SST',
#         },
#     'subject_name' : 'TS02',
#     'output_filename' : 'TS03_CRBL_passive.html'
#     }     


# session_config_list_2AFC = {
#     'list_config': [
#         session_config_TS01,
#         session_config_TS02,
#         session_config_TS06,
#         session_config_TS07,
#         session_config_TS08,
#         session_config_TS09,
#         session_config_TS03,
#         session_config_YH24,
#         ],
#     'paths' : paths,
#     } 


# session_config_list_2AFC = {
#     'list_config': [
#         session_config_TS01,
#         session_config_TS02,
#         session_config_TS06,
#         session_config_TS07,
#         session_config_TS08,
#         session_config_TS09,
#         session_config_TS03,
#         session_config_YH24,
#         ],
#     'label_names' : {
#         '-1':'Exc',
#         '1':'Inh_VIP',
#         '2':'Inh_SST',
#         },
#     'subject_name' : 'all',
#     'output_filename' : 'all_PPC_passive.html',  
#     'paths' : paths,
#     # 'pdf_spec' : pdf_spec,
#     } 




