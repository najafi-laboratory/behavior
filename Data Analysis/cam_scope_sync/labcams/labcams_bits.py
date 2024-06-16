# -*- coding: utf-8 -*-
# """
# Created on Fri Jun 14 21:03:09 2024

# @author: gtg424h
# """

# from labcams import parse_cam_log, unpackbits
# import numpy as np

# fname = 'C:/data analysis/behavior/cam_scope_sync/labcams/_cam0_run001_20240128_143427.camlog'

# # Read the log
# log,comm = parse_cam_log(fname)
# # Parse bits as timestamps of rise(onsets) and fall (offsets)
# onsets,offsets = unpackbits(log.var2)
# # Or if you prefer parse bits as lines
# bits = unpackbits(log.var2,output_binary=True)

# # plotting
# # %mat!!plot!lib inline
# import pylab as plt
# # plot bit lines
# plt.figure(figsize=[12,5])
# plt.plot(bits.T*0.5 + np.arange(bits.shape[0]),'k')
# # plot events
# for i,onkey in enumerate(onsets):
#     plt.vlines(onsets[i],onkey-0.25,onkey+0.25,color='r')
#     plt.vlines(offsets[i],onkey+.5-0.25,onkey+.5+0.25,color='b')
    
# plt.yticks(np.arange(bits.shape[0]),['ch {0}'.format(i) for i in np.arange(bits.shape[0])]);
# plt.xlabel('Frame number');
# plt.xlim([0,10000]);

from labcams import parse_cam_log,unpackbits

files = 'C:/data analysis/behavior/cam_scope_sync/labcams/_cam0_run001_20240128_143427.camlog'

sync,log = parse_cam_log(files)

unpacked_binary = unpackbits(sync['var2'].values,output_binary=True) # this is just for plotting

%matplotlib notebook
import pylab as plt
import numpy as np

plt.plot(unpacked_binary.T+np.arange(unpacked_binary.shape[0]))


onsets,offsets = unpackbits(sync['var2'].values)



[(k,len(onsets[k])) for k in onsets.keys()]
for k in onsets.keys():
    plt.plot(onsets[k],np.ones_like(onsets[k])*int(k)+1,'ko',markerfacecolor = 'none')

# to get the start frame id for each trial (assuming trials on port 2)
frames_where_the_pulse_was_first_seen = sync.frame_id[onsets[2]].values