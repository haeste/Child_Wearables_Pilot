# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 15:47:44 2023

@author: nct76
"""
import glob
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from scipy.signal import butter, lfilter, freqz
import scipy
import pandas as pd
import numpy as np
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
fs = 1/30.0       # sample rate, Hz
high = 0.000555555556
low = 0.000138888889

subjects = []
acc_m = {}
gyro_m = {}
loc = ''
day_of_rec = []
time_accurate =[]
part = []
durations = []
rates = []
variances = []
periods = []
var2 = []


for subj in subjects:
    nappa_acc_files = glob.glob(loc +subj+'/Napping_Pants/'+ '*Acc*.csv')
    for f in nappa_acc_files:
        print(f)
        accn = pd.read_csv(f)
        accn['Time'] = accn['UTCTimestamp(ms)'].astype('datetime64[ms]')
        acc_m[subj] = accn
        
    nappa_gyro_files = glob.glob(loc +subj+'/Napping_Pants/'+ '*Gyro*.csv')
    for f in nappa_gyro_files:
        print(f)
        gyro = pd.read_csv(f)
        gyro['Time'] = gyro['UTCTimestamp(ms)'].astype('datetime64[ms]')
        if (gyro.Time[len(gyro.Time)-1].day - gyro.Time[0].day) > 1:
            
            print('2 nights')
        time_in = sum(gyro[' feature3_Y(unitless)']>0.35)/len(gyro)
        day = gyro.Time[0].day
        part.append(subj)
        day_of_rec.append(day)
        time_accurate.append(time_in)
        durations.append((gyro.Time[len(gyro.Time)-1] - gyro.Time[0]))
        invalid = gyro[' feature3_Y(unitless)']<0.35
        gyro.loc[invalid, ' feature4_Y(Hz)'] = np.nan
        
        gyro_m[subj] = gyro[gyro[' feature3_Y(unitless)']>=0.35]
        #gyro_m[subj] = gyro
        rates.append(np.mean(gyro_m[subj][' feature4_Y(Hz)']))
        variances.append(np.std(gyro_m[subj][' feature4_Y(Hz)']))
        #plt.plot(gyro_m[subj]['Time'],gyro_m[subj][' feature3_Y(unitless)'])
        
        
        gyro[' feature4_Y(Hz)'] = gyro[' feature4_Y(Hz)'].interpolate(method='linear')
        gyro[' feature4_Y(Hz)'] = gyro[' feature4_Y(Hz)'].fillna(gyro[' feature4_Y(Hz)'].median())
        y1 = gyro[' feature4_Y(Hz)']
        y1 = (y1-np.mean(y1))/np.std(y1)

        y1 = butter_lowpass_filter(y1, low, fs, 1, high=high)
        #y1 = y1[60:-60]
        t  = gyro.Time[60:-60]
        peaks, _ = scipy.signal.find_peaks(y1, prominence=0.65)
        period = (np.mean(np.diff(peaks))*30)/60
        periods.append(period)
        #plt.plot(t,y1)
        #plt.plot(t[peaks+60], y1[peaks])
        n=len(y1)
        k=np.arange(n)
        T=n/fs
        frq=k/T
        frq = frq[:len(frq)//2]
        Y = np.fft.fft(y1)/n
        Y = Y[:int(n/2)]
        plt.plot(frq,abs(Y))
        var2.append(np.std(y1))
df = pd.DataFrame({'subj': part, 'day': day_of_rec, 'time_acc':time_accurate, 'duration': durations, 'rate':rates, 'variances':variances, 'periods':periods, 'var2':var2})


import seaborn as sns
plt.figure()
sns.stripplot(data=df,x='Participant',y='rate', hue='Participant')
plt.ylabel('Resp Rate (bpm)')
plt.figure()
sns.stripplot(data=df,x='Participant',y='variances', hue='Participant')
plt.ylabel('Resp Rate Variance (bpm)')
plt.figure()
sns.stripplot(data=df,x='Participant',y='periods', hue='Participant')
plt.ylabel('Period (m)')
plt.xlabel('Participant')