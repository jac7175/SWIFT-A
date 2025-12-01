import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import butter, sosfiltfilt, sosfreqz
import functions as func  # import custom functions

####### --- Insert file path here --- #######
filePath = '/path/to/022624_191850.h5'
####### ----------------------------- #######

timeName = filePath[-16:-3] # extract time from file name
keys = ['GeoU', 'HydN'] # sensors to process

data = func.readH5FilesData(filePath) # read in h5 file
fs, N, dt, times = func.getDataParams(filePath) # extract basic signal processing info

y1 = data[keys[0]] # break out time series for each sensor
y2 = data[keys[1]] # break out time series for each sensor

# Filtering geophone signal
cut_low = 4.5  # low cut frequency [Hz]
cut_high = 90  # high cut frequency [Hz]
filt_order = 5 # filter order
sos = butter(filt_order, [cut_low, cut_high], btype='bandpass', fs=fs, output='sos')
y1 = sosfiltfilt(sos, y1)  # filtfilt to ensure no phase distortion

cohDict = func.cohGram(y1, y2, 2**14, fs, numWinPerCOH=6, overlap=0.1) # calculate coherencegram
Gxy_avg, Gxy_mtx, freqsGxy, timesWin, df_win = func.gxy(y1, y2, 0.25, 'hann', 2 ** 15, fs, type='gxy') # calculate cross power spectrum

# Extract relevant variables from cohDict
coh = cohDict['coh']
freqs = cohDict['freqs']
timesCoh = cohDict['timesCoh']
coh_avg = np.mean(coh,axis=1) # calculate average coherence
Gxy_db = 10*np.log10(np.abs(Gxy_mtx)) # convert cross power spectrum to dB

# Use Computer Modern font
font = {'family': 'serif',
        'size': 16,
        'serif': 'cmr10'
        }
plt.rc('font', **font)
plt.rcParams.update({
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True
})

# Plots
fig = plt.figure(figsize=(12, 6)) #figsize=(12, 6)
gs = gridspec.GridSpec(
    2, 2,
    figure=fig,
    height_ratios=[3, 1],  # left column only
    width_ratios=[7, 1]
)
ax1 = fig.add_subplot(gs[0, 0])  # coherence spectrogram
ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)  # avg coherence (shrinks with cbar)
ax3 = fig.add_subplot(gs[1, 0], sharex=ax1)  # time series

c = ax1.imshow(
    coh, cmap='viridis',
    extent=[timesCoh[0], timesCoh[-1], min(freqs), max(freqs)],
    origin='lower', aspect='auto'
)
cbar = fig.colorbar(
    c, ax=ax1,
    orientation='horizontal',
    pad=0.05, shrink=0.7, location='top'
)
# cbar.set_label(r"Coherence, $\gamma^2$")
cbar.set_label(r"Coherence, $\gamma^2$", rotation=0, labelpad=10)
cbar.ax.xaxis.set_label_position('top')
cbar.ax.xaxis.set_label_coords(1.125, 0.4)
c.set_clim(0, 1)
ax1.set_ylabel('Frequency [Hz]')
# ax1.set_ylim([0, 5000])
ax1.set_ylim([1, 1000])
ax1.set_yscale('log')
ax1.set_xlim([timesCoh[0], timesCoh[-1]])
ax1.tick_params(axis='x', labelbottom=False)
ax2.plot(coh_avg, freqs, 'k')
ax2.set_xlabel(r"Avg. coh., $\overline{\gamma^2}$")
ax2.set_ylim([1, 1000])
ax2.set_xlim([0, 1])
ax2.tick_params(axis='y', labelleft=False)
plt.tight_layout()
pos2 = ax2.get_position()
new_height = pos2.height * 0.80
ax2.set_position([pos2.x0, pos2.y0, pos2.width, new_height])

ax3.plot(times, y1, 'k')
ax3.set_ylabel("Velocity [m/s]")
ax3.set_xlabel("Time [s]")
ax3.set_xlim([times[0], times[-1]])
pos3 = ax3.get_position()
ax3.set_position([pos3.x0, pos3.y0 + 0.08, pos3.width, pos3.height])
ax3.text(0.015, 0.90, f"{keys[0]}", transform=ax3.transAxes,
         fontsize=10, ha='left', va='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=1))
plt.show()


fig, ax1 = plt.subplots(figsize=(12, 6))
c = ax1.imshow(Gxy_db, cmap='viridis',
    extent=[timesWin[0], timesWin[-1], min(freqsGxy), max(freqsGxy)],
    origin='lower', aspect='auto')
cbar = fig.colorbar(c, ax=ax1, orientation='horizontal',
                    pad=0.05, shrink=0.7, location='top')
cbar.set_label(r"Cross power, [dB re: 1 (m/s)Pa/Hz]")
c.set_clim(-120, -60)
ax1.set_ylabel('Frequency [Hz]')
ax1.set_ylim([1, 1000])
ax1.set_yscale('log')
ax1.set_xlabel('Time [s]')
ax1.set_xlim([timesWin[0], timesWin[-1]])
plt.tight_layout()
plt.show()

print('Done.')