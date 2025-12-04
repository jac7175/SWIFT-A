import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from scipy.signal import butter, sosfiltfilt
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
cut_low = 4.5
cut_high = 90
filt_order = 5
sos = butter(filt_order, [cut_low, cut_high], btype='bandpass', fs=fs, output='sos')
y1 = sosfiltfilt(sos, y1)  # filtfilt to ensure no phase distortion
y2 = sosfiltfilt(sos, y2)  # filtfilt to ensure no phase distortion

cohDict = func.cohGram(y1, y2, 2**14, fs, numWinPerCOH=6, overlap=0.1)
Gxy_avg, Gxy_mtx, freqsGxy, timesWin, df_win = func.gxy(y1, y2, 0.25, 'hann', 2 ** 15, fs, type='gxy')

freqs = cohDict['freqs']
cut_low_idx = np.argmin(np.abs(freqs - cut_low))
cut_high_idx = np.argmin(np.abs(freqs - cut_high))
freqs = freqs[cut_low_idx:cut_high_idx]
coh = cohDict['coh'][cut_low_idx:cut_high_idx]

timesCoh = cohDict['timesCoh']
coh_avg = np.mean(coh, axis=1)  # avg over time: function of freq
coh_avg_time = np.mean(coh, axis=0)  # avg over freq: function of time
Gxy_db = 10*np.log10(np.abs(Gxy_mtx))

frac_st_time = 7
frac_ed_time = 12
frac_st_idx = np.argmin(np.abs(timesCoh - frac_st_time))
frac_ed_idx = np.argmin(np.abs(timesCoh - frac_ed_time))
coh_avg_frac = np.mean(coh[:,frac_st_idx:frac_ed_idx],axis=1)

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


fig = plt.figure(figsize=(15, 6))
outer = gridspec.GridSpec(
    2, 2,
    width_ratios=[1, 1.3],
    height_ratios=[3, 1],
    wspace=0.25,
    hspace=0.25
)
fig.subplots_adjust(
    left=0.06,
    right=1.0,
    top=0.93,
    bottom=0.09,
    wspace=0.25,
    hspace=0.25
)
# TOP-LEFT: Gxy spectrogram  --> ax1
ax1 = fig.add_subplot(outer[0, 0])
c1 = ax1.imshow(
    Gxy_db, cmap='viridis',
    extent=[timesWin[0], timesWin[-1], min(freqsGxy), max(freqsGxy)],
    origin='lower', aspect='auto', interpolation='hanning'
)
ax1.tick_params(axis='x', labelbottom=False)
cbar1 = fig.colorbar(
    c1, ax=ax1, orientation='horizontal',
    pad=0.05, shrink=0.7, location='top'
)
cbar1.set_label(r"Cross power, [dB re: 1 (m/s)Pa/Hz]", labelpad=10)
c1.set_clim(-120, -60)
ax1.set_ylabel('Frequency [Hz]')
ax1.set_ylim([4, 100])
ax1.set_yscale('log')
ax1.set_xlim([0, 60])

# BOTTOM-LEFT: Time series  --> ax2
ax2 = fig.add_subplot(outer[1, 0])
ax2.plot(times, y1, 'k')
ax2.set_ylabel("Velocity [m/s]")
ax2.set_xlabel("Time [s]")
ax2.set_xlim([0, 60])
ax2.text(0.02, 0.90, f"{keys[0]}",
         transform=ax2.transAxes,
         fontsize=12, ha='left', va='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=1))

pos2 = ax2.get_position()
ax2.set_position([pos2.x0, pos2.y0 + 0.03, pos2.width, pos2.height])

# RIGHT COLUMN: coherence spectrogram + avg coherence + avg coh vs time
gs_right = outer[:, 1].subgridspec(
    2, 2,
    height_ratios=[3, 1],
    width_ratios=[7, 1]
)
# ax3: coherence spectrogram
ax3 = fig.add_subplot(gs_right[0, 0])

# ax4: avg coherence vs frequency
ax4 = fig.add_subplot(gs_right[0, 1], sharey=ax3)

# ax5: avg coherence vs time
ax5 = fig.add_subplot(gs_right[1, 0], sharex=ax3)

# ---- Coherence spectrogram (ax3) ----
c2 = ax3.imshow(
    coh, cmap='viridis',
    extent=[timesCoh[0], timesCoh[-1], min(freqs), max(freqs)],
    origin='lower', aspect='auto', interpolation='hanning'
)
ax3.axvline(frac_st_time, color='r', linestyle='--',linewidth=2)
ax3.axvline(frac_ed_time, color='r', linestyle='--',linewidth=2)
cbar2 = fig.colorbar(
    c2, ax=ax3,
    orientation='horizontal',
    pad=0.05, shrink=0.7, location='top'
)
cbar2.set_label(r"Coherence, $\gamma^2$", rotation=0, labelpad=10)
c2.set_clim(0, 1)

ax3.set_ylabel('Frequency [Hz]')
ax3.set_ylim([4, 100])
ax3.set_yscale('log')
ax3.set_xlim([0, 60])
ax3.tick_params(axis='x', labelbottom=False)

# ---- Avg coherence vs frequency (ax4) ----
ax4.plot(coh_avg, freqs, 'k')
ax4.plot(coh_avg_frac, freqs, 'r')
ax4.set_xlabel(r"$\langle \gamma^2 \rangle_f$")
ax4.set_ylim([4, 100])
ax4.set_yscale('log')
ax4.set_xlim([0, 1])
ax4.tick_params(axis='y', labelleft=False)
pos4 = ax4.get_position()
ax4.set_position([pos4.x0 - 0.03, pos4.y0, pos4.width, pos4.height * 0.80])

# ---- Avg coherence vs time (ax5, bottom-right) ----
ax5.plot(timesCoh, coh_avg_time, 'k')
ax5.axvline(x=frac_st_time, color='r', linestyle='--', linewidth=2)
ax5.axvline(x=frac_ed_time, color='r', linestyle='--', linewidth=2)
ax5.set_ylabel(r"$\langle \gamma^2 \rangle_t$")
ax5.set_xlabel("Time [s]")
ax5.set_ylim([0, 1])
pos5 = ax5.get_position()
ax5.set_position([pos5.x0, pos5.y0 + 0.03, pos5.width, pos5.height])

rect = patches.Rectangle(
    (frac_st_time, 0),
    frac_ed_time - frac_st_time,
    1,
    facecolor='lightcoral', # A light shade of red
    alpha=0.3,             # Transparency (0 is fully transparent, 1 is opaque)
)
ax5.add_patch(rect)

# Vertical divider line between left and right halves
fig.add_artist(plt.Line2D(
    [0.45, 0.45], [0.02, 0.98],
    transform=fig.transFigure,
    color='black',
    linewidth=1
))
plt.show()

print('Done.')
