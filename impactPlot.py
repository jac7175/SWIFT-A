import numpy as np
import time
import matplotlib.pyplot as plt
import scipy as sp
import h5py
import datetime

def readH5FilesData(filePath, data=None):
    """Reads H5 files containing impact data time series.

    This function takes in a path to an H5 file and returns a dictionary of time series data.

    Parameters:
        filePath (str): path to the H5 file.

    Returns:
        data (dict): A dictionary of time series data. The keys are the channel names and the values are the time series data.
    """
    if data is None:
        data = {}
    with h5py.File(filePath, 'r') as f:
        dataGroup = f['data']
        keys = list(dataGroup.keys())
        for key in keys:
            data[key] = np.array(np.squeeze(dataGroup[key][:]))
    return data

def applySensorSensitivity(data: dict):  # input data as a dictionary
    """Applies sensor sensitivity to time series data based on channel names and nominal sensitivities defined in the script.

    This function takes in a dictionary of time series data as created by readH5FilesData and applies the sensor sensitivity
    to each channel based on the channel name and nominal sensitivity.
    It returns a new dictionary with the channel names and their corresponding time series data after applying the sensor sensitivity.

    Parameters:
        data (dict): Dictionary of time series data. The keys are the channel names and the values are the time series data.

    Returns:
        result (dict): A dictionary of time series data after nominal sensor sensitivity is applied.
    """
    GeoSens = 28.8  # nominal geophone sensitivity [V/(m/s)]
    MicSens = 50 * 10 ** -3  # nominal mic sensitivity [V/Pa]
    HydSens = 0.006309  # nominal Hydrophone sensitivity [V/Pa]
    HammerSens = 0.23 * 10 ** -3  # nominal hammer sensitivity [V/N]

    # Map channel keys to their sensitivities
    sensitivity_mapping = {
        'HydN': HydSens,
        'HydE': HydSens,
        'HydS': HydSens,
        'HydW': HydSens,
        'MicC': MicSens,
        'MicW': MicSens,
        'MicN': MicSens,
        'MicE': MicSens,
        'GeoN': GeoSens,
        'GeoW': GeoSens,
        'GeoU': GeoSens,
        'force_hammer': HammerSens
    }
    # Check if all keys in the data dictionary have defined sensitivities
    unknown_channels = [ch for ch in data.keys() if ch not in sensitivity_mapping]
    if unknown_channels:
        raise ValueError(f"Unknown channel(s) in data: {unknown_channels}")
    # Apply sensitivity to each channel in the dictionary
    result = {
        channel: readings / sensitivity_mapping[channel]
        for channel, readings in data.items()
    }
    return result

def getSigParams(data,fs):
    """Returns standard signal processing parameters for an input time series.

    This function takes in a time series and returns basic signal processing parameters:
     sample rate, number of samples, time interval, frequency interval, and time vector.

    Parameters:
        data (list or numpy array): List or array of time series data
        fs (int): Sample rate [Hz]

    Returns:
        dt (float): Sample spacing in time [s]
        N (int): Number of samples [#]
        T (float): Total record length [s]
        df (float): Frequency spacing [Hz]
        times (numpy array): Time vector [s]
    """
    dt = 1 / fs
    data = np.array(data)
    N = len(data)
    T = N * dt
    df = 1 / T
    times = np.arange(0, N) * dt
    return dt, N, T, df, times

def rxy(timeSeries1,timeSeries2,fs):
    """Returns cross correlation between two input time series.

   This function takes in two time series and returns the cross correlation between them.

    Parameters:
        timeSeries1 (numpy array): Array of time series 1 (both time series must be the same length)
        timeSeries2 (numpy array): Array of time series 2 (both time series must be the same length)
        fs (int): Sample rate [Hz]

    Returns:
        R12 (numpy array): Cross correlation between time series 1 and 2
        tau (numpy array): Time shift corresponding to the cross correlation output
        zeroLagValue (float): Value of the cross correlation at zero lag
        zeroLagValueNorm (float): Value of the cross correlation at zero lag, normalized by the rms values of intput time series 1 and 2
        C12 (numpy array): Cross correlation between time series 1 and 2, normalized by the rms values of intput time series 1 and 2
    """
    N = np.size(timeSeries1)  # size of time series 1
    N2 = np.size(timeSeries2) # size of time series 2
    if N != N2: # check to ensure that both time series are the same length
        raise ValueError("The two input time series must have the same length.")
    dt = 1 / fs  # calculate sample spacing [s]
    T = N * dt  # total record lenght [s]
    X1 = np.fft.fft(timeSeries1, axis=0) * dt # linear spectrum of time series 1
    X2 = np.fft.fft(timeSeries2, axis=0) * dt # linear spectrum of time series 2
    S12 = np.conjugate(X1) * X2 / T # double-sided cross power spectrum between 1 and 2
    S11 = np.conjugate(X1) * X1 / T # double-sided auto power spectrum of 1
    S22 = np.conjugate(X2) * X2 / T # double-sided auto power spectrum of 2
    R12 = np.fft.ifft(S12) / dt  # cross correlation between 1 and 2
    R11 = np.fft.ifft(S11) / dt  # auto correlation of 1
    R22 = np.fft.ifft(S22) / dt  # auto correlation of 2
    C12 = R12 / (np.sqrt(R11[0] * R22[0])) # normalized cross correlation between 1 and 2
    zeroLagValue = R12[0] # zero lag value of cross correlation between 1 and 2
    zeroLagValueNorm = C12[0] # zero lag value of normalized cross correlation between 1 and 2
    tau = np.arange(0, N) * dt # lag (time) vector
    negLags = tau > (N/2) * dt
    tau[negLags] = tau[negLags] - T # rearragne to allow for negative lags
    R12 = np.concatenate((R12[negLags],R12[~negLags])) # rearrange to put in standard format (increasing lag from neg to pos)
    C12 = np.concatenate((C12[negLags], C12[~negLags])) # rearrange to put in standard format (increasing lag from neg to pos)
    tau = np.concatenate((tau[negLags],tau[~negLags])) # rearrange to put in standard format (increasing lag from neg to pos)
    return R12, tau, zeroLagValue, zeroLagValueNorm, C12


if __name__ == '__main__':
    filePath = '/path/to/sync/impact/file'
    print('Loading file...')
    start_time = time.time()             # defines start time for code timing
    data = readH5FilesData(filePath) # reads in h5 file
    y = applySensorSensitivity(data)             # apply sensor senitivities
    fs = 52100                                           # sample rate [samples/sec]
    dt, N, T, df, times = getSigParams(y['HydN'],fs)  # extract basic signal processing info

    print('Processing...')
    peaks = sp.signal.find_peaks(y['force_hammer'],height=0.5*np.max(y['force_hammer']),distance=fs*1)  # extract peaks (min dist of 0.5 sec, 1000N) from hammer channel
    hitStartIndex = peaks[0].astype('int32') - np.floor(0.01 * fs).astype('int32')  # move 0.01 second before peak to capture start index
    hitEndIndex = hitStartIndex + np.floor(0.5 * fs).astype('int32') # add half a second to start to define end index for each hit within a set of 3 hits
    numHits = len(peaks[0])   # number of hits at specified location

    zeroPad = []  # initiate zeroPad list
    for hitNum in range(len(hitStartIndex)):  # loop fills out a list that is as long as the number of hits that occur at one location
        zeroPad.append(np.zeros([hitEndIndex[hitNum] - hitStartIndex[hitNum]]))

    chsToPlot = np.arange(0,12) # all channels

    lenCorrList = numHits * len(chsToPlot) # length of concatenated list of all channels across all specified hits
    corrList = [] # initiate corrList list
    corrListLabels = []  # initiate corrListLables variable
    for ch in list(data): # This loop appends all hydrophone chanels across 3 hits to one variable for later correlation matrix processing
         for hitNum in range(numHits):
            corrList.append(np.concatenate([y[ch][hitStartIndex[hitNum]:hitEndIndex[hitNum]], zeroPad[hitNum]]))
            corrListLabels.append(ch + ', ' + 'Hit ' + str(hitNum))


    RxyZip = [[[] for _ in range(lenCorrList)] for _ in range(lenCorrList)]  # initiate RxyZip (will contain Rxy, Cxy, tau...)
    zeroLagValue = np.empty((lenCorrList,lenCorrList))   # initiate
    zeroLagValuesNorm = np.empty((lenCorrList,lenCorrList)) # initiate
    Rxy = [[[] for _ in range(lenCorrList)] for _ in range(lenCorrList)]  # initate
    Cxy = [[[] for _ in range(lenCorrList)] for _ in range(lenCorrList)]  # initiate
    for corrCount1 in range(len(corrList)):  # loop calcs Rxy, Cxy, zeroLag and time shift tau for all possible channel/hit combinations
        for corrCount2 in range(len(corrList)):
            RxyZip[corrCount1][corrCount2] = rxy(corrList[corrCount1],corrList[corrCount2],fs)  # main calc line
            Rxy[corrCount1][corrCount2] = RxyZip[corrCount1][corrCount2][0]  # extracts Rxy into its own 2d list
            zeroLagValue[corrCount1, corrCount2] = np.real(RxyZip[corrCount1][corrCount2][2]) # extracts zeroLag into its own 2d list
            zeroLagValuesNorm[corrCount1, corrCount2] = np.real(RxyZip[corrCount1][corrCount2][3]) # extracts zeroLagNorm into its own 2d list
            Cxy[corrCount1][corrCount2] = np.real(RxyZip[corrCount1][corrCount2][4]) # extracts Cxy into its own 2d list
    tau = RxyZip[0][0][1]  # lag vector (same across all hits/channels)


    plt.rcParams['font.family'] = 'sans-serif'  # Use sans-serif fonts
    plt.rcParams['font.sans-serif'] = 'Helvetica'  # use Helvetica
    #
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(4, 1, 1)
    ax2 = fig.add_subplot(4, 1, 2)
    ax3 = fig.add_subplot(4, 1, 3)
    ax4 = fig.add_subplot(4, 1, 4)
    ax1.plot(tau,Cxy[-3][12],label = '1')
    ax1.plot(tau,Cxy[-2][13],label = '2')
    ax1.plot(tau,Cxy[-1][14],label = '3')
    ax1.grid()
    ax1.set_xlim([-0.005,0.04])
    ax1.legend(ncol=3, title="Hit Index", edgecolor='k',facecolor='w',framealpha=1)
    ax1.set_ylim([-0.6,0.7])
    ax1.text(0.02, 0.90, "North Hydrophone", transform=ax1.transAxes,
            fontsize=10, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=1))
    ax2.plot(tau,Cxy[-3][15],label = corrListLabels[-3] + ': ' + corrListLabels[15])
    ax2.plot(tau,Cxy[-2][16],label = corrListLabels[-2] + ': ' + corrListLabels[16])
    ax2.plot(tau,Cxy[-1][17],label = corrListLabels[-1] + ': ' + corrListLabels[17])
    ax2.grid()
    ax2.set_xlim([-0.005,0.04])
    ax2.set_ylim([-0.6,0.7])
    ax2.text(0.02, 0.90, "South Hydrophone", transform=ax2.transAxes,
            fontsize=10, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=1))
    ax3.plot(tau,Cxy[-3][9],label = corrListLabels[-3] + ': ' + corrListLabels[9])
    ax3.plot(tau,Cxy[-2][10],label = corrListLabels[-2] + ': ' + corrListLabels[10])
    ax3.plot(tau,Cxy[-1][11],label = corrListLabels[-1] + ': ' + corrListLabels[11])
    ax3.grid()
    ax3.set_xlim([-0.005,0.04])
    ax3.set_ylim([-0.6,0.7])
    ax3.text(0.02, 0.90, "East Hydrophone", transform=ax3.transAxes,
            fontsize=10, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=1))
    ax4.plot(tau,Cxy[-3][18],label = corrListLabels[-3] + ': ' + corrListLabels[18])
    ax4.plot(tau,Cxy[-2][19],label = corrListLabels[-2] + ': ' + corrListLabels[19])
    ax4.plot(tau,Cxy[-1][20],label = corrListLabels[-1] + ': ' + corrListLabels[20])
    ax4.grid()
    ax4.set_xlim([-0.005,0.04])
    ax4.set_xlabel('Lag  [s]')
    ax4.set_ylim([-0.6,0.7])
    ax4.text(0.02, 0.90, "West Hydrophone", transform=ax4.transAxes,
            fontsize=10, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=1))
    fig.text(0, 0.5, 'Normalized Cross Correlation', va='center', rotation='vertical')
    plt.tight_layout()
    fig.show()

    print(f'Done. Completed in {np.round(time.time() - start_time,2)} seconds')