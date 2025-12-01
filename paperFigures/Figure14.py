import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import time
import os
from itertools import combinations
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import functions as func  # import custom functions

####### --- Insert file path here --- #######
filePath = '/path/to/C8.h5'
####### ----------------------------- #######

print('Loading file...')
start_time = time.time()  # defines start time for code timing
data = func.readH5FilesData(filePath)  # reads in h5 file›
del data['dateTime']
y = data
fs = 51200  # sample rate [samples/sec]
dt, N, T, df, times = func.getSigParams(y['HydN'], fs)  # extract basic signal processing info

print('Processing...')
peaks = sp.signal.find_peaks(y['force_hammer'], height=0.5 * np.max(y['force_hammer']), distance=fs * 1)  # extract peaks (min dist of 0.5 sec, 1000N) from hammer channel
hitStartIndex = peaks[0].astype('int32') - np.floor(0.01 * fs).astype('int32')  # move 0.01 second before peak to capture start index
hitEndIndex = hitStartIndex + np.floor(0.5 * fs).astype('int32')  # add half a second to start to define end index for each hit within a set of 3 hits
numHits = len(peaks[0])  # number of hits at specified location

zeroPad = []  # initiate zeroPad list
for hitNum in range(len(hitStartIndex)):  # loop fills out a list that is as long as the number of hits that occur at one location
    zeroPad.append(np.zeros([hitEndIndex[hitNum] - hitStartIndex[hitNum]]))

lenCorrList = numHits * len(y)  # length of concatenated list of all channels across all specified hits
corrList = {}  # initiate corrList list
for ch in list(data):  # This loop appends all hydrophone chanels across 3 hits to one variable for later correlation matrix processing
    for hitNum in range(numHits):
        label = ch + ', ' + 'Hit ' + str(hitNum)
        corrList[label] = {}
        corrList[label]['time_series'] = np.concatenate([y[ch][hitStartIndex[hitNum]:hitEndIndex[hitNum]], zeroPad[hitNum]])
        corrList[label]['Gxy'] = func.gxy(y[ch][hitStartIndex[hitNum]:hitEndIndex[hitNum]], y[ch][hitStartIndex[hitNum]:hitEndIndex[hitNum]], 0, 'hann', 2**13, fs, type='gxx')

for key1 in list(corrList.keys()):  # loop calcs Rxy, Cxy, zeroLag and time shift tau for all possible channel/hit combinations
    for key2 in list(corrList.keys()):
        corrList[key1][key2] = {}
        Rxy_temp = func.rxy(corrList[key1]['time_series'], corrList[key2]['time_series'], fs)
        tau = Rxy_temp[1]
        corrList[key1][key2]['Rxy'] = Rxy_temp[0]
        corrList[key1][key2]['zero_lag'] = np.real(Rxy_temp[2])
        corrList[key1][key2]['zero_lag_norm'] = np.real(Rxy_temp[3])
        corrList[key1][key2]['Cxy'] = np.real(Rxy_temp[4])

        corrList[key1][key2]['Cxy_max_index'] = np.argmax(corrList[key1][key2]['Cxy'])
        corrList[key1][key2]['Cxy_max_value'] = corrList[key1][key2]['Cxy'][corrList[key1][key2]['Cxy_max_index']]
        corrList[key1][key2]['peak_time_diff'] = tau[corrList[key1][key2]['Cxy_max_index']]

pltTitle = 'Location: ' + os.path.splitext(os.path.basename(filePath))[0]

# Time difference of arrival (DOA) values
tdoas = np.array([
    tau[corrList['HydS, Hit 0']['HydN, Hit 0']['Cxy_max_index']], # 12
    tau[corrList['HydE, Hit 0']['HydN, Hit 0']['Cxy_max_index']], # 13
    tau[corrList['HydW, Hit 0']['HydN, Hit 0']['Cxy_max_index']], # 14
    tau[corrList['HydE, Hit 0']['HydS, Hit 0']['Cxy_max_index']], # 23
    tau[corrList['HydW, Hit 0']['HydS, Hit 0']['Cxy_max_index']], # 24
    tau[corrList['HydW, Hit 0']['HydE, Hit 0']['Cxy_max_index']], # 34
])

# 4 sensor positions (meters)
sensor_positions = np.array([
    [367.03, -160.926, -74.93],  # N
    [-367.03, 160.926, -74.93],  # S
    [160.926, 367.03, 74.93],    # E
    [-160.926, -367.03, 74.93]   # W
]) * 10**-3

M = len(sensor_positions)   # number of sensors
K = int(M*(M-1)/2)          # number of unique combinations of sensors
unique_pairs = list(combinations(range(M), 2))  # list of unique pairs
X = []                      # matrix of sensor location differences
T = []                      # matrix of time shifts

for ii in range(K):
    jj, kk = unique_pairs[ii]
    X.append(sensor_positions[kk, :] - sensor_positions[jj, :])
    T.append(tdoas[ii])
X = np.array(X)
T = np.array(T)

S, residuals, rank, s = np.linalg.lstsq(X, T, rcond=None)  # u = least square soln, res = sum a squared residuals,
norm_S = np.linalg.norm(S)
c_est1 = np.sqrt(1/(S[0]**2 + S[1]**2 + S[2]**2)) # should be the same as below
c_est2 = 1/norm_S # should be the same as above
azimuth_deg = np.degrees(np.arctan2(S[1], S[0]))

# Set up plot formatting to use Computer Modern font
font = {'family': 'serif',
        'size': 16,
        'serif': 'cmr10'
        }
plt.rc('font', **font)
plt.rcParams.update({
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True
})

fig, ax1 = plt.subplots(figsize=(12, 6))
# fig.suptitle(pltTitle + ', Hydrophone - Hydrophone, Hit 0', fontsize=14)
ax1.plot(tau, corrList['HydN, Hit 0']['HydN, Hit 0']['Cxy'], label='N-N')
ax1.plot(tau, corrList['HydN, Hit 0']['HydS, Hit 0']['Cxy'], label='N-S')
ax1.plot(tau, corrList['HydN, Hit 0']['HydE, Hit 0']['Cxy'], label='N-E')
ax1.plot(tau, corrList['HydN, Hit 0']['HydW, Hit 0']['Cxy'], label='N-W')

for k in ['HydN, Hit 0','HydS, Hit 0','HydE, Hit 0','HydW, Hit 0']:
    ax1.plot(
        tau[corrList['HydN, Hit 0'][k]['Cxy_max_index']],
        corrList['HydN, Hit 0'][k]['Cxy_max_value'],
        'o', markersize=10, color='red'
    )
ax1.legend()
ax1.set_xlim([-0.005, 0.02])
ax1.set_xlabel('Lag [s]')
ax1.set_ylabel('Normalized Cross Correlation')
ax2 = inset_axes(
    ax1,
    width="130%",
    height="130%",
    bbox_to_anchor=(0.60, 0.55, 0.3, 0.3),  # << move left/down here
    bbox_transform=ax1.transAxes,
    axes_class=plt.PolarAxes,
    loc='upper right')
azimuth_rad = np.radians(azimuth_deg)
ax2.set_theta_zero_location('N')
ax2.set_theta_direction(-1)
ax2.arrow(
    azimuth_rad, 0,
    0, 1.0,
    width=0.03,
    head_width=0.1,
    head_length=0.1,
    color='red')
ax2.set_rmax(1.2)
ax2.set_rticks([])
angles = [0, 45, 90, 135, 180, 225, 270, 315]
ax2.set_thetagrids(angles, labels=[rf"{a}$^\circ$" for a in angles])
title_str = rf"DOA = {azimuth_deg+180 + 180:.1f}$^\circ$"
ax2.set_title(title_str, va='bottom')
plt.tight_layout()
plt.show()
