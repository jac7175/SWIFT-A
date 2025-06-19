import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import datetime
import functions


filePath = '/path/to/030224_105346.h5'

key = 'HydN'
numRecs = 12

data = functions.readH5FilesData(filePath)
dateTime = datetime.datetime.strftime(data['dateTime'], '%m%d%y_%H%M%S')
dateTimeFileNameString = dateTime + '.h5'
dataTimeSeries = data[key]
fs, N, dt, times = functions.getDataParams(filePath)

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
    Gxy_avg, _, freqs, _, _ = functions.gxy(timeRec, timeRec, 0.5, 'hann', 2**13, fs, type='gxx')
    k = sp.stats.kurtosis(timeRec)
    k = np.round(k)
    envPdfOut = functions.env_pdf(timeRec, fs, num_bins=500)
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