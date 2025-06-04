import numpy as np
import time
import matplotlib.pyplot as plt
import scipy as sp
import h5py
import datetime


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

def readH5FilesData(filePath, ch2read='all', data=None):
    if data is None:
        data = {}
    with h5py.File(filePath, 'r') as f:
        dataGroup = f['data']
        cdt = str(f['datetime/'][()])[2:-1]
        creationDateTime = datetime.datetime.strptime(cdt, "%Y-%m-%dT%H:%M:%S.%fUTC")
        keys = list(dataGroup.keys())
        if ch2read == 'all':
            for key in keys:
                data[key] = np.array(np.squeeze(dataGroup[key][:]))
        else:
            for key in ch2read:
                data[key] = np.array(np.squeeze(dataGroup[key][:]))
        data['dateTime'] = creationDateTime
    return data

def getDataParams(filePath):
    f = h5py.File(filePath, 'r')
    fs = f['data'].attrs['sample_rate']
    N = len(f['data']['HydN'])
    dt = 1/fs
    times = np.arange(0, N) * dt
    return fs, N, dt, times

def gxy(timeSeries1, timeSeries2, overlap, window, nFFT, fs, type='gxx'):
    """Calculates the cross-power spectrum of two input time series
        Parameters:
            timeSeries1: time series 1d array
            timeSeries2: time series 1d array
            overlap: overlap between subsequent blocks, range 0-1
            window: window, 'rectangular', 'hann', 'flattop', 'hamming'
            nFFT: number of points in FFT window
            fs: sampling rate, samples/second
    """
    dt = 1 / fs
    timeSeries1 = np.array(timeSeries1)
    timeSeries2 = np.array(timeSeries2)
    N1 = np.size(timeSeries1)
    N2 = np.size(timeSeries2)
    if N1 > N2:
        N = N2
        timeSeries1 = timeSeries1[0:N]
    elif N2 > N1:
        N = N1
        timeSeries2 = timeSeries1[0:N]
    else:
        N = N1
    nOverlap = int(nFFT * overlap)
    nAdv = nFFT - nOverlap
    nWins = int(np.floor((N - nFFT) / nAdv) + 1)
    timesWin = np.arange(0, nWins) * nAdv / fs + nFFT / 2 / fs
    T_win = nFFT * dt
    df_win = 1 / T_win
    freqs = np.arange(0, nFFT / 2) * df_win

    if window == 'rectangular':
        w = np.ones(nFFT)
    elif window == 'hann':
        w = np.hanning(nFFT)
    elif window == 'flattop':
        w = sp.signal.windows.flattop(nFFT)
    elif window == 'hamming':
        w = np.hamming(nFFT)
    else:
        raise ValueError("No window function defined.")

    GxyTemp = []
    for wIndex in np.arange(0, nWins):
        # print(f'Processing {wIndex} of {nWins}')
        advInx = wIndex * nAdv
        sig1 = timeSeries1[advInx:nFFT + advInx] * w / np.mean(w ** 2)  # may need a ,0 in the indexing for ice2024 data
        sig2 = timeSeries2[advInx:nFFT + advInx] * w / np.mean(w ** 2)
        lnspc1 = np.fft.fft(sig1, axis=0) * dt
        if type == 'gxx':
            lnspc2 = lnspc1
        elif type == 'gxy':
            lnspc2 = np.fft.fft(sig2, axis=0) * dt
        GxyTemp.append(2 / T_win * np.conjugate(lnspc1[0:int(nFFT / 2)]) * lnspc2[0:int(nFFT / 2)])
    Gxy_avg = np.sum(GxyTemp, axis=0) / nWins
    Gxy_mtx = np.rot90(GxyTemp)
    return Gxy_avg, Gxy_mtx, freqs, timesWin, df_win

def hilbertXform(linear_spectrum,N,fs):
    dt = 1/fs
    neg_freq = np.zeros(int(N/2-1))
    f0 = np.ones(1)
    pos_freq = 2*np.ones(int(N/2-1))
    fs_2 = np.ones(1)
    weight = np.concatenate((f0,pos_freq,fs_2,neg_freq))
    linear_spectrum_xformed = linear_spectrum * weight
    complex_time_series = np.fft.ifft(linear_spectrum_xformed)/dt
    return linear_spectrum_xformed, complex_time_series

def lin_spectrum(time_series,fs):
    dt, N, T, df, times = getSigParams(time_series, fs)
    linear_spectrum = np.fft.fft(time_series,axis=0)*dt
    return linear_spectrum

def env_pdf(time_series, fs, num_bins=5000):
    N = len(time_series)
    X = lin_spectrum(time_series, fs)
    X_h, x_h = hilbertXform(X, N, fs)
    env_x = np.abs(x_h)
    pdf_env, bins = np.histogram(env_x, bins=num_bins, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2  # Get the midpoints of bins
    return env_x, pdf_env, bin_centers