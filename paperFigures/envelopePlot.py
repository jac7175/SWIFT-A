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
pdf_fit = []
pdf_fit_x = []
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

v_wind = [0, 5, 10, 20]



for ii in range(numRecs):
    timeRec = dataTimeSeries[startInd:endInd]
    Gxy_avg, _, freqs, _, df_win = functions.gxy(timeRec, timeRec, 0.5, 'hann', 2**13, fs, type='gxx')
    k = sp.stats.kurtosis(timeRec)
    k = np.round(k)
    envPdfOut = functions.env_pdf(timeRec, fs, num_bins=500)
    env_x.append(envPdfOut[0])
    pdf_env.append(envPdfOut[1])
    bin_centers.append(envPdfOut[2])

    fit_x = np.geomspace(10**-4, np.max(envPdfOut[0]), 1000)
    pdf_fit_env_noise_sig_params = sp.stats.rayleigh.fit(envPdfOut[0])
    pdf_fit_env_noise_sig_loc = pdf_fit_env_noise_sig_params[0]
    pdf_fit_env_noise_sig_scale = pdf_fit_env_noise_sig_params[1]
    pdf_env_noise_sig_fit = sp.stats.rayleigh.pdf(fit_x, loc=pdf_fit_env_noise_sig_loc,
                                                  scale=pdf_fit_env_noise_sig_scale)
    pdf_fit.append(pdf_env_noise_sig_fit)
    pdf_fit_x.append(fit_x)

    ax1.plot(bin_centers[ii], pdf_env[ii], label=f'{int(k)}', color=cmap(ii))
    ax2.plot(times[startInd:endInd], timeRec, label=f'{int(k)}', color=cmap(ii))
    ax3.plot(freqs, 10 * np.log10(np.abs(Gxy_avg) * 10**6), label=f'{int(k)}', color=cmap(ii))
    startInd += ptsPerRec
    endInd += ptsPerRec

NL_wind = np.zeros(len(freqs))
for jj in range(len(freqs)):
    NL_wind[jj] = (41.2 + 10 * np.log10(np.pi) + 22.4 * np.log10(1)- 10*np.log10(1.5+(freqs[jj]/1000)**1.59))

ax1.plot(pdf_fit_x[0], pdf_fit[0], color='k', label='Rayleigh\nfit',linestyle='dashed')
ax1.set_xlabel("Magnitude: $|x|$\n(b)")
ax1.set_ylabel('Probability Density')
ax1.set_xlim([0.01, 10])
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylim([10 ** -3, 10 ** 2])
ax2.set_xlabel("Time [s]\n(a)")
ax2.set_ylabel('Amplitude [Pa]')
ax2.set_xlim([0, 60])
ax2.set_ylim([-5, 5])
ax3.plot(freqs, NL_wind, linestyle='dotted', color='k')
ax3.annotate('1 m/s wind noise', xy=(16**3, 32), xytext=(16**3, 32),fontsize=10,rotation=-10)
ax3.set_xlabel("Frequency [Hz]\n(c)")
ax3.set_ylabel(r"Power Spectral Density [dB re μPa$^2$/Hz]")
ax3.set_xlim([10, 20000])
# ax3.set_ylim([-100, -10])
ax3.set_xscale('log')
ax1.legend(bbox_to_anchor=(1.02, 1.03), ncol=1, prop={'family': 'monospace'}, title='Kurtosis',
           title_fontproperties={'size': 10})
plt.tight_layout()
plt.show()

print('Done.')