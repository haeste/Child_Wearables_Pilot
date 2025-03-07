# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 15:32:54 2024

@author: nct76
"""
import pandas as pd
import more_itertools as mit
import numpy as np
import pyedflib
import matplotlib.pyplot as plt
def loadEDF(file_name):
    f = pyedflib.EdfReader(file_name)
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    
    ecg = f.readSignal(0)
    accX = f.readSignal(1)
    accY = f.readSignal(2)
    accZ = f.readSignal(3)
    hrv = f.readSignal(5)
    
    ecg_sf = f.getSampleFrequencies()[0]
    accX_sf = f.getSampleFrequencies()[1]
    accY_sf = f.getSampleFrequencies()[2]
    accZ_sf = f.getSampleFrequencies()[3]
    hrv_sf = f.getSampleFrequencies()[5]
    
    startdate = f.getStartdatetime()
    
    ecg_t = np.arange(1,len(ecg)+1)/ecg_sf
    accX_t = np.arange(1,len(accX)+1)/accX_sf
    accY_t = np.arange(1,len(accY)+1)/accY_sf
    accZ_t = np.arange(1,len(accZ)+1)/accZ_sf
    hrv_t = np.arange(1,len(hrv)+1)/hrv_sf
    
    
    hrv_t_ms = hrv_t*1000
    hrv_t_ms =np.datetime64(startdate).astype('datetime64[ms]') + hrv_t_ms.astype(int).astype('timedelta64[ms]')
    
    accX_t_ms = accX_t*1000
    accX_t_ms =np.datetime64(startdate).astype('datetime64[ms]') + accX_t_ms.astype(int).astype('timedelta64[ms]')
    
    accY_t_ms = accY_t*1000
    accY_t_ms =np.datetime64(startdate).astype('datetime64[ms]') + accY_t_ms.astype(int).astype('timedelta64[ms]')
    
    accZ_t_ms = accZ_t*1000
    accZ_t_ms =np.datetime64(startdate).astype('datetime64[ms]') + accZ_t_ms.astype(int).astype('timedelta64[ms]')
    
    ecg_t_ms = ecg_t*1000
    ecg_t_ms =np.datetime64(startdate).astype('datetime64[ms]') + ecg_t_ms.astype(int).astype('timedelta64[ms]')
    return ecg,ecg_t_ms,ecg_sf, hrv, hrv_t_ms,hrv_sf, accX, accX_t_ms,accX_sf, accY, accY_t_ms,accY_sf, accZ, accZ_t_ms,accZ_sf

def getRRIntWithinPhysRange(intervals,sf,calc_len = 2):
    
    hrv_chunkedlen = int(calc_len * (sf*60))
    
    rrs_err = [np.array(x)[(np.array(x)<250) | (np.array(x)>1200)] for x in mit.chunked(intervals, hrv_chunkedlen)]
    rrs = [np.array(x)[(np.array(x)>250) & (np.array(x)<1200)] for x in mit.chunked(intervals, hrv_chunkedlen)]
    return rrs_err, rrs
def getRRnotDetected(intervals,sf,calc_len = 2):
    
    hrv_chunkedlen = calc_len * (sf*60)
    
    rrs_err = [np.array(x)[(np.array(x)>1200)] for x in mit.chunked(intervals, hrv_chunkedlen)]
    rdff_t = np.arange(0,len(rrs_err))*(60 *calc_len)
    np.array([sum(r==65534) for r in rrs_err])
    return rrs_err, rdff_t

def computeENMO(x):
    
    r = np.sqrt(np.sum(np.power(np.array(x),2),axis=1))

    return r

def getcontiguous(wt):
    for a_i, a in enumerate(wt):
        if a_i < 3:
            contiguous = np.median(wt[:a_i+3])
        elif a_i >(len(wt)-3):
            contiguous = np.median(wt[a_i-3:]) 
        else:
            contiguous = np.median(wt[a_i-3:a_i+3])
              
        if a and not contiguous:
            wt[a_i] = contiguous
    return wt

WEARTIME_THRESHOLD = 9.5

# Load in the ECG files
bittiumfiles = [] # Specify list of path names to the ECG data
subj_i = 1 # Specify subject ID to analyse
ecg,ecg_t_ms,ecg_sf, hrv, hrv_t_ms,hrv_sf, accX, accX_t_ms,accX_sf, accY, accY_t_ms,accY_sf, accZ, accZ_t_ms,accZ_sf = loadEDF(bittiumfiles[subj_i])
acc_sig = np.array([accX,accY,accZ]).T
acc_t_ms = accX_t_ms

duration_seconds = 60*10
duration_in_steps = int(((duration_seconds)*1000)/(acc_t_ms[1] - acc_t_ms[0]).astype(float))

#Calculate ENMO on acceleration measured by Bittium
enmol = computeENMO(acc_sig)
enmol = abs(enmol)-np.median(abs(enmol))
enmo = np.array([np.mean(x) for x in mit.chunked(enmol, duration_in_steps)])
enmo = np.array([np.mean(abs(x - np.mean(x))) for x in mit.chunked(enmol, duration_in_steps)])

# For automated weartime calculation, check enmo against threshold
auto_weartime = enmo>WEARTIME_THRESHOLD


auto_weartime = getcontiguous(auto_weartime)
startdate = acc_t_ms[0].astype('datetime64[s]')
acc_t_min =np.arange(np.datetime64(startdate).astype('datetime64[s]'),np.datetime64(startdate).astype('datetime64[s]') + np.timedelta64(duration_seconds).astype('timedelta64[s]')*len(enmo), duration_seconds)

# Filter out RR intervals outwith the physiological range for infants.
rrs_err, rrs = getRRIntWithinPhysRange(hrv,hrv_sf)
rrs_hr_err, rrs_hr = getRRIntWithinPhysRange(hrv,hrv_sf,calc_len=1)

# Estimate noiseyness of signal as the count of non-physiological RR intervals
noise = np.mean(np.array([sum(r>0) for r in rrs_err]))/2
# Calculate heart rate 
hr = np.array([60*1000*(1/np.mean(r)) if len(r)>5 else np.nan for r in rrs_hr])
# Calculate the SDNN 
hrv_calc = np.array([np.std(r) if len(r)>100 else np.nan for r in rrs])
rdff_t = np.arange(0,len(hrv_calc))*(60 *2)
hr_t = np.arange(0,len(hr))*(60 *1)

# Calculate mean and standard deviation of HRV
std_hrv = np.nanstd(hrv_calc)
mean_hrv = np.nanmean(hrv_calc)

hrv_zscore = (hrv_calc - mean_hrv)/std_hrv

std_hr = np.nanstd(hr)
mean_hr = np.nanmean(hr)

hr_zscore = (hr - mean_hr)/std_hr

rdff_t_ms = rdff_t*1000
rdff_hr_t_ms = hr_t*1000
rdff_t_ms =np.datetime64(startdate).astype('datetime64[ms]') + rdff_t_ms.astype(int).astype('timedelta64[ms]')
rdff_hr_t_ms =np.datetime64(startdate).astype('datetime64[ms]') + rdff_hr_t_ms.astype(int).astype('timedelta64[ms]')
#%% Plot HR and HRV
plt.plot(rdff_t_ms,hrv_zscore,label='HRV - SDRRI')
hr[hr==0]= np.nan
plt.plot(rdff_hr_t_ms,hr_zscore, label='HR')
plt.ylabel('(s/bpm)')
plt.legend()
plt.show()



#%% Plot the HR and HRV for first nap

nap1start= np.datetime64('1970-01-01 12:25:00')
nap1stop= np.datetime64('1970-01-01 14:15:00')
during_nap = (rdff_t_ms>nap1start) & (rdff_t_ms<nap1stop)
t_hrv =rdff_t_ms[during_nap]-rdff_t_ms[during_nap][0]
t_hrv = t_hrv.astype('timedelta64[m]')
plt.plot(t_hrv,hrv_zscore[during_nap],label='HRV - SDRRI', color='b')
hr[hr==0]= np.nan
during_nap = (rdff_hr_t_ms>nap1start) & (rdff_hr_t_ms<nap1stop)
t = rdff_hr_t_ms[during_nap]- rdff_hr_t_ms[during_nap][0]
t = t.astype('timedelta64[m]')
plt.plot(t,hr_zscore[during_nap], label='HR', color='r')
plt.legend()
plt.ylabel('z-score')
plt.xlabel('Time (minutes)')
plt.show()


#%% Plot the HR and HRV for second nap

nap2start= np.datetime64('1970-01-01 12:25:00')
nap2stop= np.datetime64('1970-01-01 14:15:00')
during_nap = (rdff_t_ms>nap2start) & (rdff_t_ms<nap2stop)
t_hrv =rdff_t_ms[during_nap]-rdff_t_ms[during_nap][0]
t_hrv = t_hrv.astype('timedelta64[m]')
plt.plot(t_hrv,hrv_zscore[during_nap], label='HRV - SDRRI', color='b')
hr[hr==0]= np.nan
during_nap = (rdff_hr_t_ms>nap2start) & (rdff_hr_t_ms<nap2stop)
t = rdff_hr_t_ms[during_nap]- rdff_hr_t_ms[during_nap][0]
t = t.astype('timedelta64[m]')
plt.plot(t,hr_zscore[during_nap], label='HR', color='r')
plt.legend()

plt.ylabel('z-score')
plt.xlabel('Time (minutes)')
plt.show()

#%%
nap3start= np.datetime64('1970-01-01 17:30:00')
nap3stop= np.datetime64('1970-01-01 18:30:00')

during_nap = (rdff_t_ms>nap3start) & (rdff_t_ms<nap3stop)
t_hrv =rdff_t_ms[during_nap]-rdff_t_ms[during_nap][0]
t_hrv = t_hrv.astype('timedelta64[m]')
plt.plot(t_hrv,hrv_zscore[during_nap], color='b')
hr[hr==0]= np.nan
during_nap = (rdff_hr_t_ms>nap3start) & (rdff_hr_t_ms<nap3stop)
t = rdff_hr_t_ms[during_nap]- rdff_hr_t_ms[during_nap][0]
t = t.astype('timedelta64[m]')
plt.plot(t,hr_zscore[during_nap], color='r')
plt.ylabel('z-score')
plt.xlabel('Time (minutes)')
plt.show()

#%% Plot the first sleep HR and HRV
sleepstart1 = np.datetime64('1970-01-01 19:08:00')
sleepstop1 = np.datetime64('1970-01-01 07:40:00')
during_sleep = (rdff_hr_t_ms>sleepstart1) & (rdff_hr_t_ms<sleepstop1)
t = rdff_hr_t_ms[during_sleep]- rdff_hr_t_ms[during_sleep][0]
t = t.astype('timedelta64[m]')
plt.plot(t,hr_zscore[during_sleep], color='k')

#%%
from scipy.signal import butter, lfilter, freqz
import pandas as pd
import matplotlib.pyplot as plt
def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)
def butter_highpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='high', analog=False)

def butter_bandpass(cutoff, fs, order=2):
    return butter(order, cutoff, fs=fs, btype='bandpass', analog=False)

def butter_lowpass_filter(data, low, fs, order=5, high=None):
    if high != None:
        b, a = butter_bandpass([low, high],fs, order=order)
        y = lfilter(b, a, data)
    else:
        b, a = butter_lowpass(low, fs, order=order)
        y = lfilter(b, a, data)
    
    return y

order = 3
fs = 30.0       # sample rate, Hz
high = 0.000555555556
low = 0.000138888889

# Load accelerometry and gyroscope data from Napping Pants for current subject

# Day 1
accn = pd.read_csv('SN00_AccFeatures_(00-00-00_00.00.00).csv')
accn['Time'] = accn['UTCTimestamp(ms)'].astype('datetime64[ms]')
gyro = pd.read_csv('SN00_GyroFeatures_(00-00-00_00.00.00).csv')
gyro['Time'] = gyro['UTCTimestamp(ms)'].astype('datetime64[ms]')
# Day 2
accn2 = pd.read_csv('SN00_AccFeatures_(00-00-00_00.00.01).csv')
accn2['Time'] = accn2['UTCTimestamp(ms)'].astype('datetime64[ms]')
gyro2 = pd.read_csv('SN00_GyroFeatures_(00-00-00_00.00.01).csv')
gyro2['Time'] = gyro2['UTCTimestamp(ms)'].astype('datetime64[ms]')
# Day 3
accn3 = pd.read_csv('SN00_AccFeatures_(00-00-00_00.00.02).csv')
accn3['Time'] = accn3['UTCTimestamp(ms)'].astype('datetime64[ms]')
gyro3 = pd.read_csv('SN00_AccFeatures_(00-00-00_00.00.02).csv')
gyro3['Time'] = gyro3['UTCTimestamp(ms)'].astype('datetime64[ms]')

t = gyro['Time']
fs = 1/(t[1] - t[0]).seconds

# Plot raw data for each day
ax1 = plt.subplot(4,1,1)
plt.plot(acc_t_min,enmo, label='Bittium')
plt.plot(accn.Time, accn[' feature2(m/sec)'], label='NAPPA', color='r')
plt.plot(accn2.Time, accn2[' feature2(m/sec)'], color='r')
plt.plot(accn3.Time, accn3[' feature2(m/sec)'], color='r')
plt.legend()
plt.ylabel('Acc (g)')
#%% Plot HRV for each day
ax2 = plt.subplot(412, sharex=ax1)
plt.plot(rdff_t_ms,hrv_calc,label='HRV - SDRRI')
hr[hr==0]= np.nan
plt.plot(rdff_hr_t_ms,hr, label='HR')
plt.ylabel('(s/bpm)')
plt.legend()
#%% Remove respiration rates we are not confident in
invalid = gyro[' feature3_Y(unitless)']<0.35
gyro.loc[invalid, ' feature4_Y(Hz)'] = np.nan
invalid2 = gyro2[' feature3_Y(unitless)']<0.35
gyro2.loc[invalid2, ' feature4_Y(Hz)'] = np.nan
invalid3 = gyro3[' feature3_Y(unitless)']<0.35
gyro3.loc[invalid3, ' feature4_Y(Hz)'] = np.nan
#%% Plot Respiration rates
ax3 = plt.subplot(413, sharex=ax2)
plt.plot(gyro.Time, gyro[' feature4_Y(Hz)'], 'r--')
plt.plot(gyro2.Time, gyro2[' feature4_Y(Hz)'], 'r--')
plt.plot(gyro3.Time, gyro3[' feature4_Y(Hz)'], 'r--')
plt.ylabel('Respiration (bpm)')
# Plot NAPPA Movement
ax4 = plt.subplot(414, sharex=ax3)
plt.plot(accn.Time, accn[' feature1(class N)'], color='r')
plt.plot(accn2.Time, accn2[' feature1(class N)'], color='r')
plt.plot(accn3.Time, accn3[' feature1(class N)'], color='r')
#Plot NAPPA position
plt.ylabel('Position')
ax4.set_yticks([1,2,3,4,5, 6])
ax4.set_yticklabels(['Left', 'Right','Supine','Prone','Down', 'Up'])


#%%