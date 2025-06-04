import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import datetime
import h5py


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

def getSigParams(data,fs):
    dt = 1 / fs
    data = np.array(data)
    N = len(data)
    T = N * dt
    df = 1 / T
    times = np.arange(0, N) * dt
    return dt, N, T, df, times


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

if __name__ == '__main__':
    filePath = 'path/to/030224_105346.h5'

    key = 'HydN'
    numRecs = 12

    data = readH5FilesData(filePath)
    dateTime = datetime.datetime.strftime(data['dateTime'], '%m%d%y_%H%M%S')
    dateTimeFileNameString = dateTime + '.h5'
    dataTimeSeries = data[key]
    fs, N, dt, times = getDataParams(filePath)

    ptsPerRec = int(np.floor(N / numRecs))
    env_x = []
    pdf_env = []
    bin_centers = []
    startInd = 0
    endInd = ptsPerRec
    cmap = plt.get_cmap('tab20c')

    fig = plt.figure(figsize=(10, 7))
    ax2 = fig.add_subplot(2, 2, 1)
    ax1 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 1, 2)

    plt.rcParams['font.family'] = 'sans-serif'  # Use sans-serif fonts
    plt.rcParams['font.sans-serif'] = 'Helvetica'  # use Helvetica

    for ii in range(numRecs):
        timeRec = dataTimeSeries[startInd:endInd]
        Gxy_avg, _, freqs, _, _ = gxy(timeRec, timeRec, 0.5, 'hann', 2**13, fs, type='gxx')
        k = sp.stats.kurtosis(timeRec)
        k = np.round(k)
        envPdfOut = env_pdf(timeRec, fs, num_bins=500)
        env_x.append(envPdfOut[0])
        pdf_env.append(envPdfOut[1])
        bin_centers.append(envPdfOut[2])
        ax1.plot(bin_centers[ii], pdf_env[ii], label=f'{int(k)}', color=cmap(ii))
        ax2.plot(times[startInd:endInd], timeRec, label=f'{int(k)}', color=cmap(ii))
        ax3.plot(freqs, 10 * np.log10(np.abs(Gxy_avg)), label=f'{int(k)}', color=cmap(ii))
        startInd += ptsPerRec
        endInd += ptsPerRec

    ax1.set_xlabel("Magnitude: $|x|$\n(b)")
    ax1.set_ylabel('Probability Density')
    ax1.set_xlim([0.01, 10])
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylim([10 ** -3, 10 ** 2])
    # ax1.set_title(r"PDF of Envelope " + dateTime)
    # ax2.set_title('Time series')
    ax2.set_xlabel("Time [s]\n(a)")
    ax2.set_ylabel('Amplitude [Pa]')
    ax2.set_xlim([0, 60])
    ax2.set_ylim([-5, 5])
    # ax3.set_title('Power Spectrum')
    ax3.set_xlabel("Frequency [Hz]\n(c)")
    ax3.set_ylabel('Power Spectrum [Pa$^2$/Hz]')
    ax3.set_xlim([10, 20000])
    ax3.set_ylim([-100, -10])
    ax3.set_xscale('log')
    plt.tight_layout()
    ax1.legend(loc='upper right', ncol=3, prop={'family': 'monospace'}, title='Kurtosis', title_fontproperties={'size':10})
    plt.show()

    print('Done.')