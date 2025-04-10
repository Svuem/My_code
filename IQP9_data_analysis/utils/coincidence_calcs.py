'''
@author: Jonas Rauchfuß
@mailto: jrauchfu@physnet.uni-hamburg.de

Created on 27.07.2023
'''
import os
import sys

import numpy as np
import h5py
from tqdm import tqdm
from numba import jit

def compute_coincidence_times(hdf5file: str, detector_0: int = 0, detector_1: int = 1, detector_2: int = 2, verbose: bool = False) -> None:
    """
    Computes the coincidence times between three photon detectors and stores the results in an HDF5 file.

    Parameters:
    hdf5file (str): Path to the HDF5 file containing photon detection data.
    detector_0 (int): Index of the first detector (0, 1, or 2).
    detector_1 (int): Index of the second detector (0, 1, or 2).
    detector_2 (int): Index of the third detector (0, 1, or 2).
    verbose (bool): If True, print additional information about the computation. Default is False.

    Returns:
    None: The function modifies the HDF5 file in place to store the coincidence times.
    """
    with h5py.File(hdf5file, 'r+') as hf:
        # Read the datasets from the HDF5 file
        data = hf['data']
        timestamps_ps = np.array(data['timestamps_ps'], dtype='int64')
        channels = np.array(data['channels'])

        # Calculate total counts for each detector
        total_counts_detector_0 = (channels == detector_0).sum()
        total_counts_detector_1 = (channels == detector_1).sum()
        total_counts_detector_2 = (channels == detector_2).sum()

        # Initialize a 3D array to store coincidence times, filled with NaN
        coinc_times = np.empty((3, 3, len(timestamps_ps)))  # Note: Fixed the shape for 3 detectors total
        coinc_times[:] = np.nan

        print('Calculating Coincidence Times...')

        # Create an iterable for processing (with progress bar if verbose)
        if verbose:
            iteration_range = tqdm(range(len(channels)-2))
        else:
            iteration_range = range(len(channels)-2)

        # Loop through the channels to calculate coincidence times
        for i in iteration_range:
            # Skip if the current channel is the same as the next one
            if channels[i] == channels[i+1]:
                continue

            # Calculate the time difference between successive clicks from different detectors
            coinc_times[channels[i], channels[i+1], i] = timestamps_ps[i+1] - timestamps_ps[i]

            # Additionally check the third channel for coincidence
            if channels[i+2] != channels[i] and channels[i+2] != channels[i+1]:
                coinc_times[channels[i], channels[i+2], i] = timestamps_ps[i+2] - timestamps_ps[i]

        for i in iteration_range:
            # Skip if the current channel is the same as the next one
            if channels[i] == channels[i-1]:
                continue

            # Calculate the time difference between successive clicks from different detectors
            coinc_times[channels[i-1], channels[i], i] = timestamps_ps[i] - timestamps_ps[i-1]

            # Additionally check the third channel for coincidence
            if channels[i-2] != channels[i] and channels[i-2] != channels[i-1]:
                coinc_times[channels[i-2], channels[i], i] = timestamps_ps[i] - timestamps_ps[i-2]

        # Delete any existing coincidences group if present
        if 'coincidences' in hf:
            del hf['coincidences']

        # Create a new group and dataset for storing coincidence times
        coinc_group = hf.create_group('coincidences')
        coinc_dset = coinc_group.create_dataset('coincidence_times_ps',
                                                shape=np.shape(coinc_times),
                                                dtype='float',
                                                data=coinc_times)

        # Store total counts and measurement time as attributes
        coinc_dset.attrs.create('total_counts_detector_0', data=total_counts_detector_0)
        coinc_dset.attrs.create('total_counts_detector_1', data=total_counts_detector_1)
        coinc_dset.attrs.create('total_counts_detector_2', data=total_counts_detector_2) # Fixed typo in attribute name
        coinc_dset.attrs.create('total_measurement_time_ps', data=timestamps_ps[-1])

        # Verbose output of total measurement time and total coincidences counted
        if verbose:
            print(f'  Total Measurement time was: {timestamps_ps[-1]} ps')
            # print(f'  Total possible coincidences counted: {np.sum(len(timestamps_ps) * 9 - np.isnan(coinc_times))}\n')
            print('All done, ready to continue.')
# def compute_coincidence_times(hdf5file: str, verbose: bool = False) -> None:
#     '''
#     Finds possible coincidences in a dataset and computes their respective time difference.
#     '''
#     with h5py.File(hdf5file, 'r+') as hf:
#         data = hf['data']
#         timestamps_ps = np.array(data['timestamps_ps'], dtype='int64')
#         channels = np.array(data['channels'])

#         total_counts_detector_0 = (channels == 0).sum()
#         total_counts_detector_1 = (channels == 1).sum()

#         coinc_times = []

#         if verbose:
#             iteration_range = tqdm(range(len(channels)-1))
#         else:
#             iteration_range = range(len(channels)-1)

#         for i in iteration_range:
#             if (channels[i] + channels[i+1]) % 2:
#                 coinc_times.append(timestamps_ps[i+1] - timestamps_ps[i])

#         # if 'coincidences' in hf:
#         #     del hf['coincidences']
#         coinc_group = hf.create_group('coincidences')
#         coinc_dset = coinc_group.create_dataset('coincidence_time',
#                                     shape=np.shape(coinc_times),
#                                     dtype='int64',
#                                     data=coinc_times)

#         coinc_dset.attrs.create('total_counts_detector_0', data=total_counts_detector_0)
#         coinc_dset.attrs.create('total_counts_detector_1', data=total_counts_detector_1)
#         coinc_dset.attrs.create('total_measurement_time_ps', data=timestamps_ps[-1])

#         if verbose:
#             print(f'  Total Measurement time was: {timestamps_ps[-1]} ps')
#             print(f'  Total possible coincidences counted: {len(coinc_times)}\n')


def count_coincidences(hdf5file, coincidence_window_width):
    '''

    '''
    with h5py.File(hdf5file) as hf:
        times = hf['coincidences/coincidence_time']

        # TODO: Comment
        # coinc_num = (times[:] <= coincidence_window_width/2).sum()
        coinc_num = int(np.sum(np.logical_or((np.abs(times[:])<coincidence_window_width/2),
                                            (np.abs(times[:])<coincidence_window_width/2)))
                                            /2) # all coincidences are counted twice in the above calculation, therefore we divide the number by 2


        counts_detector_0 = hf['coincidences/coincidence_time'].attrs.get('total_counts_detector_0')
        counts_detector_1 = hf['coincidences/coincidence_time'].attrs.get('total_counts_detector_1')
        measurement_time = hf['coincidences/coincidence_time'].attrs.get('total_measurement_time_ps')

        counts_detector_0 = float(counts_detector_0)
        counts_detector_1 = float(counts_detector_1)
        measurement_time = float(measurement_time*1e-12)

        coinc_rate = coinc_num / measurement_time

        acc_coinc_rate = float(counts_detector_0/measurement_time
                               * counts_detector_1/measurement_time
                               * coincidence_window_width*1e-12)

    return coinc_num, coinc_rate, acc_coinc_rate

def convert_count_number_to_count_rate(hdf5file, count_number):
    with h5py.File(hdf5file) as hf:
        measurement_time_ps = hf['coincidences/coincidence_times_ps'].attrs.get('total_measurement_time_ps')
    measurement_time = float(measurement_time_ps*1e-12)
    return count_number/measurement_time

def get_accidental_count_rate(hdf5file, coincidence_window_ps, detector_a=0, detector_b=1):
    with h5py.File(hdf5file) as hf:
        counts_detector_a = hf['coincidences/coincidence_times_ps'].attrs.get(f'total_counts_detector_{detector_a}')
        counts_detector_b = hf['coincidences/coincidence_times_ps'].attrs.get(f'total_counts_detector_{detector_b}')
        measurement_time = hf['coincidences/coincidence_times_ps'].attrs.get('total_measurement_time_ps')

    measurement_time = float(measurement_time*1e-12)

    acc_coinc_rate = float(counts_detector_a/measurement_time
                            * counts_detector_b/measurement_time
                            * coincidence_window_ps*1e-12)
    return acc_coinc_rate

def count_double_coincidences(hdf5file, coincidence_window_ps, detector_a=0, detector_b=1, tau=0):
    '''
    Counts the total number of coincidences between detector_a and
    detector_b within the conincidence_window (given in ps).
    Optionally applies a time shift tau on the counts of detector_b.

    ARGS:
        hdf5file: str
            filepath of the measurement data containing the
            coincidence_times mdataset that is created using
            compute_coincidence_times().
        detector_a: int
            index of detector_a
        detector_b: int
            index of detector_b (with time delay tau)
        couincidence_window_ps: int
            full coincidence window in picoseconds
        tau: int
            time delay for the timestamps of detector_b

    RETURNS:
        int: total number of coincidences

    RAISES:
        KeyError: When the coincidence_times dataset does not exist in
        the measurement data
    '''

    with h5py.File(hdf5file) as hf:
        try:
            coincidence_times = np.array(hf['coincidences']['coincidence_times_ps'])
        except KeyError:
            raise KeyError('Could not load coincidence_times dataset. Make sure the dataset exists.')

    number_of_coincidences = int(np.sum(np.logical_or((np.abs(coincidence_times[detector_b,detector_a,:]-tau)<coincidence_window_ps/2),
                                                      (np.abs(coincidence_times[detector_a,detector_b,:]+tau)<coincidence_window_ps/2)))
                                                      /2) # all coincidences are counted twice in the above calculation, therefore we divide the number by 2

    return number_of_coincidences


def count_triple_coincidences(hdf5file, coincidence_window_ps, detector_a = 0, detector_b = 1, detector_c = 2, tau = 0):
    '''
    Counts the total number of coincidences between detector_a,
    detector_b and detector_c within the conincidence_window (given in ps).
    Optionally applies a time shift tau on the counts of detector_c.

    ARGS:
        hdf5file: str
            filepath of the measurement data containing the
            coincidence_times mdataset that is created using
            compute_coincidence_times().
        detector_a: int
            index of detector_a
        detector_b: int
            index of detector_b
        detector_c: int
            index of detector_c (with time delay tau)
        couincidence_window_ps: int
            full coincidence window in picoseconds
        tau: int
            time delay for the timestamps of detector_b

    RETURNS:
        int: total number of coincidences

    RAISES:
        KeyError: When the coincidence_times dataset does not exist in
        the measurement data
    '''

    with h5py.File(hdf5file) as hf:
        try:
            coincidence_times = np.array(hf['coincidences']['coincidence_times_ps'])
        except KeyError:
            raise KeyError('Could not load coincidence_times dataset. Make sure the dataset exists.')

    number_of_coincidences = int(np.sum(np.logical_and(np.logical_or(np.logical_or(np.abs(coincidence_times[detector_c,detector_a,:]-tau)<coincidence_window_ps/2,
                                                                                   np.abs(coincidence_times[detector_a,detector_c,:]+tau)<coincidence_window_ps/2),
                                                                    np.logical_or((np.abs(coincidence_times[detector_b,detector_a,:])<coincidence_window_ps/2),
                                                                                  (np.abs(coincidence_times[detector_a,detector_b,:])<coincidence_window_ps/2))),
                                                       np.logical_or((np.abs(coincidence_times[detector_b,detector_c,:]+tau)<coincidence_window_ps/2),
                                                                     (np.abs(coincidence_times[detector_c,detector_b,:]-tau)<coincidence_window_ps/2))))
                                /2)
    return number_of_coincidences


@jit(nopython=True)
def g2_unheralded(lead, lag, time, tau_arr, tau_bin_size, coincidence_window=5000):

    total_time = time[time > 0][-1]

    g2 = []

    for tau in tau_arr:

        coincidence_counter = 0

        for i, c in enumerate(lead[1:-1]):
            # i is the previous index

            # Skip if there is no event
            if c==0:
                continue

            # time of the event
            current_time = time[i+1]

            # time of the previous and next event
            previous_time = 0
            next_time = total_time+2*coincidence_window

            # check if there is an event in the lag channel one time step before or afer the current event
            if lag[i] == 1:
                previous_time = time[i]
            if lag[i+2] == 1:
                next_time = time[i+2]

            # time difference between the events
            dt_prev = previous_time - current_time
            dt_next = next_time - current_time

            # check if events occur in the coincidence window
            if abs(dt_next) <= coincidence_window:
                # check if events occur around time tau
                if  tau - tau_bin_size/2 <= dt_next <= tau + tau_bin_size/2:
                    coincidence_counter += 1
            elif abs(dt_prev) <= coincidence_window:
                 # check if events occur around time tau
                if tau - tau_bin_size/2 <= dt_prev <= tau + tau_bin_size/2:
                    coincidence_counter += 1

        g2.append( coincidence_counter / ( lead.sum() * lag.sum() ) * total_time / coincidence_window * coincidence_window / tau_bin_size)

    return g2

def g2_na(timestamps_ps, channels):
    pass

def g2_heralded(timestamps_ps, channels, tau_arr, coincidence_window=5000):

    for tau in tau_arr:
        pass



if __name__ == '__main__':
    h5file = sys.argv[1]

    compute_coincidence_times(h5file)
