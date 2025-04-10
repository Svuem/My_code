'''
@author: Jonas Rauchfuß
@mailto: jrauchfu@physnet.uni-hamburg.de

Created on 27.07.2023
'''
import io
import os

import numpy as np
import h5py


from . import read_ptu

def convert_ptu(ptufile, txtfile, hdf5file):

    read_ptu.read_ptu(ptufile, txtfile)
    print('\n.txt file created successfully!')

    with io.open(txtfile, 'r', encoding='utf-16-le') as tf:

        channels = []
        timestamps_ps = []
        for line in tf:
            if 'CHN' in line:
                channels.append(line.split()[2])
                timestamps_ps.append(line.split()[3])

    channels = np.array(channels, dtype='uint8')
    timestamps_ps = np.array(timestamps_ps, dtype='uint64')

    with h5py.File(hdf5file, 'w') as hf:
        data_group = hf.create_group('data')
        data_group.create_dataset('channels',
                                    shape=np.shape(channels),
                                    dtype='uint8',
                                    data = channels)
        data_group.create_dataset('timestamps_ps',
                                    shape=np.shape(timestamps_ps),
                                    dtype='uint64',
                                    data = timestamps_ps)
    print('.h5 file created successfully')

    os.remove(txtfile)

    print('Ready to move on!')
