import numpy as np
import time
import matplotlib.pyplot as plt
import scipy as sp
import functions


filePath = '/path/to/sync/file/C3.h5'


print('Loading file...')
start_time = time.time()             # defines start time for code timing
data = functions.readH5FilesData(filePath) # reads in h5 file
del data['dateTime']    # remove un-needed dateTime key from dictionary
y = functions.applySensorSensitivity(data)             # apply sensor senitivities
fs = 51200                                           # sample rate [samples/sec]
dt, N, T, df, times = functions.getSigParams(y['HydN'],fs)  # extract basic signal processing info

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
        RxyZip[corrCount1][corrCount2] = functions.rxy(corrList[corrCount1],corrList[corrCount2],fs)  # main calc line
        Rxy[corrCount1][corrCount2] = RxyZip[corrCount1][corrCount2][0]  # extracts Rxy into its own 2d list
        zeroLagValue[corrCount1, corrCount2] = np.real(RxyZip[corrCount1][corrCount2][2]) # extracts zeroLag into its own 2d list
        zeroLagValuesNorm[corrCount1, corrCount2] = np.real(RxyZip[corrCount1][corrCount2][3]) # extracts zeroLagNorm into its own 2d list
        Cxy[corrCount1][corrCount2] = np.real(RxyZip[corrCount1][corrCount2][4]) # extracts Cxy into its own 2d list
tau = RxyZip[0][0][1]  # lag vector (same across all hits/channels)


plt.rcParams['font.family'] = 'sans-serif'  # Use sans-serif fonts
plt.rcParams['font.sans-serif'] = 'Helvetica'  # use Helvetica
#
fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(4, 2, 1)
ax2 = fig.add_subplot(4, 2, 3)
ax3 = fig.add_subplot(4, 2, 5)
ax4 = fig.add_subplot(4, 2, 7)
ax1.plot(tau,Cxy[-3][12],label = '1')
ax1.plot(tau,Cxy[-2][13],label = '2')
ax1.plot(tau,Cxy[-1][14],label = '3')
ax1.grid()
ax1.set_xlim([-0.005,0.04])
# ax1.legend(ncol=3, title="Hit Index", edgecolor='k',facecolor='w',framealpha=1)
ax1.set_ylim([-0.6,0.7])
ax1.text(0.02, 0.90, "North Hydrophone", transform=ax1.transAxes,
        fontsize=10, ha='left', va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=1))
ax2.plot(tau,Cxy[-3][15])
ax2.plot(tau,Cxy[-2][16])
ax2.plot(tau,Cxy[-1][17])
ax2.grid()
ax2.set_xlim([-0.005,0.04])
ax2.set_ylim([-0.6,0.7])
ax2.text(0.02, 0.90, "South Hydrophone", transform=ax2.transAxes,
        fontsize=10, ha='left', va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=1))
ax3.plot(tau,Cxy[-3][9])
ax3.plot(tau,Cxy[-2][10])
ax3.plot(tau,Cxy[-1][11])
ax3.grid()
ax3.set_xlim([-0.005,0.04])
ax3.set_ylim([-0.6,0.7])
ax3.text(0.02, 0.90, "East Hydrophone", transform=ax3.transAxes,
        fontsize=10, ha='left', va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=1))
ax4.plot(tau,Cxy[-3][18])
ax4.plot(tau,Cxy[-2][19])
ax4.plot(tau,Cxy[-1][20])
ax4.grid()
ax4.set_xlim([-0.005,0.04])
ax4.set_xlabel('Lag  [s]')
ax4.set_ylim([-0.6,0.7])
ax4.text(0.02, 0.90, "West Hydrophone", transform=ax4.transAxes,
        fontsize=10, ha='left', va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=1))
fig.text(0, 0.5, 'Normalized Cross Correlation', va='center', rotation='vertical')

ax5 = fig.add_subplot(4, 2, 2)
ax6 = fig.add_subplot(4, 2, 4)
ax7 = fig.add_subplot(4, 2, 6)
ax8 = fig.add_subplot(4, 2, 8)
ax5.plot(tau, Cxy[-3][21], label='1')
ax5.plot(tau, Cxy[-2][22], label='2')
ax5.plot(tau, Cxy[-1][23], label='3')
ax5.grid()
ax5.set_xlim([-0.005, 0.04])
ax5.legend(ncol=3, title="Hit Index", edgecolor='k', facecolor='w', framealpha=1)
ax5.set_ylim([-0.4, 0.4])
ax5.text(0.02, 0.90, "Center Microphone", transform=ax5.transAxes,
         fontsize=10, ha='left', va='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=1))
ax6.plot(tau, Cxy[-3][24])
ax6.plot(tau, Cxy[-2][25])
ax6.plot(tau, Cxy[-1][26])
ax6.grid()
ax6.set_xlim([-0.005, 0.04])
ax6.set_ylim([-0.4, 0.4])
ax6.text(0.02, 0.90, "East Microphone", transform=ax6.transAxes,
         fontsize=10, ha='left', va='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=1))
ax7.plot(tau, Cxy[-3][27])
ax7.plot(tau, Cxy[-2][28])
ax7.plot(tau, Cxy[-1][29])
ax7.grid()
ax7.set_xlim([-0.005, 0.04])
ax7.set_ylim([-0.4, 0.4])
ax7.text(0.02, 0.90, "North Microphone", transform=ax7.transAxes,
         fontsize=10, ha='left', va='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=1))
ax8.plot(tau, Cxy[-3][30])
ax8.plot(tau, Cxy[-2][31])
ax8.plot(tau, Cxy[-1][32])
ax8.grid()
ax8.set_xlim([-0.005, 0.04])
ax8.set_xlabel('Lag  [s]')
ax8.set_ylim([-0.4, 0.4])
ax8.text(0.02, 0.90, "West Microphone", transform=ax8.transAxes,
         fontsize=10, ha='left', va='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=1))
fig.text(0, 0.5, 'Normalized Cross Correlation', va='center', rotation='vertical')
plt.tight_layout()
plt.show()

print(f'Done. Completed in {np.round(time.time() - start_time,2)} seconds')