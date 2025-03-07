# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 09:52:17 2023

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
    
    hrv_chunkedlen = calc_len * (sf*60)
    
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

# Any weartime without any neighbouring weartime is likely to be a mistake
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
MAX_RR_INTERVAL = 65534
NOISE_THESHOLD_RR = 150
bittiumfiles = ['sample.EDF']
#%%
weartimes_all = pd.DataFrame()
# For the number of subjects
for i in range(0,1):
    subj_i = i
    #Load the bittium data for given subject
    ecg,ecg_t_ms,ecg_sf, hrv, hrv_t_ms,hrv_sf, accX, accX_t_ms,accX_sf, accY, accY_t_ms,accY_sf, accZ, accZ_t_ms,accZ_sf = loadEDF(bittiumfiles[subj_i])
    acc_sig = np.array([accX,accY,accZ]).T
    acc_t_ms = accX_t_ms

    
    duration_seconds = 600 # Duration of the each chunk we break our data into
    duration_in_steps = int(((duration_seconds)*1000)/(acc_t_ms[1] - acc_t_ms[0]).astype(float))

    #Calculate ENMO - the median, break into chunks
    enmol = computeENMO(acc_sig)
    enmol = abs(enmol)-np.median(abs(enmol))
    #enmo = np.array([np.mean(x) for x in mit.chunked(enmol, duration_in_steps)])
    enmo = np.array([np.mean(abs(x - np.mean(x))) for x in mit.chunked(enmol, duration_in_steps)])
    auto_weartime = enmo>WEARTIME_THRESHOLD
    
    
    
    startdate = acc_t_ms[0].astype('datetime64[s]')
    acc_t_min =np.arange(np.datetime64(startdate).astype('datetime64[s]'),np.datetime64(startdate).astype('datetime64[s]') + np.timedelta64(duration_seconds).astype('timedelta64[s]')*len(enmo), duration_seconds)

    # Filter the RR intervals for physiological range
    rrs_err, rrs = getRRIntWithinPhysRange(hrv,hrv_sf)
    # For heart rate we use a chunk length of 1 minute
    rrs_hr_err, rrs_hr = getRRIntWithinPhysRange(hrv,hrv_sf,calc_len=1)

    # Estimate noise as number of non-physiological intervals
    noise = np.mean(np.array([sum(r>0) for r in rrs_err]))/2
    # Estimate average heart rate
    hr = np.array([60*1000*(1/np.mean(r)) if len(r)>5 else np.nan for r in rrs_hr])
    # Estimate average heart rate variability
    hrv_calc = np.array([np.std(r) if len(r)>100 else np.nan for r in rrs])
    rdff_t = np.arange(0,len(hrv_calc))*(60 *2)
    hr_t = np.arange(0,len(hr))*(60 *1)
    
    # For wear detection we use a chunk length of 10 minute
    rrs_err, rrs = getRRIntWithinPhysRange(hrv,hrv_sf, calc_len=10)
    # If no MAX_INTERVALS are detected, the device is not worn
    # If any of these are detected in a chunk, then the device is not worn for that chunk
    notworn = np.array([sum(r==MAX_RR_INTERVAL) for r in rrs_err])>0
    # If non-physiological intervals are detected, the device is atleast detecting noise
    # If more than NOISE_THESHOLD_RR of these are detected in a chunk, 
    # then device worn but not properly attached.
    detached = np.array([sum(r>0) for r in rrs_err])>NOISE_THESHOLD_RR
    
    # Accelerometry based measure of weartime
    auto_weartime = getcontiguous(auto_weartime)
    
    # ECG based measure of weartime
    auto_weartime_attached =  ((detached==0) & (notworn==0))
    auto_weartime_attached = getcontiguous(auto_weartime_attached)
    
    # Combine, if either accelerometry or ECG indicates worn, then worn
    auto_weartime_all = (auto_weartime) | (auto_weartime_attached)
    # Combine, if accelerometry indicates worn but ECG indicates detatch, then worn but detached
    auto_weartime_detached = (auto_weartime_all) & (~auto_weartime_attached)


    idx = np.concatenate(([0],np.flatnonzero(auto_weartime_attached[:-1]!=auto_weartime_attached[1:])+1,[auto_weartime_attached.size]))
    out_attached = list(zip(auto_weartime_attached[idx[:-1]],np.diff(idx)))
    

    starttime = acc_t_ms[0]
    weartimesOn = []
    weartimesOff = []
    weartimesAttached = []
    weartimesDur = []
    weartimes = pd.DataFrame()
    #Calculate the time spent while device attached for each block of attachment
    for i_o, o in enumerate(out_attached):
        if o[0]:
            weartimesOn.append(acc_t_ms[0] + pd.Timedelta(int(idx[i_o]*10),'minutes'))
            weartimesOff.append(acc_t_ms[0] + pd.Timedelta(int(idx[i_o+1]*10),'minutes'))
            weartimesDur.append(pd.Timedelta(int(idx[i_o+1]*10),'minutes') - pd.Timedelta(int(idx[i_o]*10),'minutes'))
            weartimesAttached.append('Yes')
            print(pd.Timedelta(int(idx[i_o+1]*10),'minutes') - pd.Timedelta(int(idx[i_o]*10),'minutes'))

    
    idx = np.concatenate(([0],np.flatnonzero(auto_weartime_detached[:-1]!=auto_weartime_detached[1:])+1,[auto_weartime_detached.size]))
    out_dettached = list(zip(auto_weartime_detached[idx[:-1]],np.diff(idx)))
    
    #Calculate the time spent while device dettached, for each block of detechment
    for i_o, o in enumerate(out_dettached):
        if o[0]:
            weartimesOn.append(acc_t_ms[0] + pd.Timedelta(int(idx[i_o]*10),'minutes'))
            weartimesOff.append(acc_t_ms[0] + pd.Timedelta(int(idx[i_o+1]*10),'minutes'))
            weartimesDur.append(pd.Timedelta(int(idx[i_o+1]*10),'minutes') - pd.Timedelta(int(idx[i_o]*10),'minutes'))
            weartimesAttached.append('No')
    
    weartimes['On'] = weartimesOn
    weartimes['Off'] = weartimesOff
    weartimes['Attached']= weartimesAttached
    weartimes['duration'] = weartimesDur
    weartimes['subj'] = i
    weartimes = weartimes[weartimes.duration>pd.Timedelta(30, 'minutes')]
    
    errors = []
    hrv_list = []
    hrv_prop_calc = []
    hr_list = []
    hr_prop_calc = []
    
    print('Subject: ' + str(subj_i))
    print(str(len(weartimes[weartimes.Attached=='Yes'].On)) + ' wear times, total duration: ' + str(weartimes[weartimes.Attached=='Yes'].duration.sum()))
    print(str(len(weartimes[weartimes.Attached=='No'].On)) + ' dettachments, with a total duration of: ' + str(weartimes[weartimes.Attached=='No'].duration.sum()))

    # Calculate the error rate based on number of non-physiological RR intervals 
    # Calculate only for chunks when the device is considered worn and attached
    for index, wt in weartimes.iterrows():
        rrs_err, rrs = getRRIntWithinPhysRange(hrv[(hrv_t_ms>wt['On']) & (hrv_t_ms<wt['Off'])],hrv_sf)
        rrs_hr_err, rrs_hr = getRRIntWithinPhysRange(hrv[(hrv_t_ms>wt['On']) & (hrv_t_ms<wt['Off'])],hrv_sf,calc_len=1)
        noise = np.mean(np.array([sum(r>0) for r in rrs_err]))/2
        hr = np.array([60*1000*(1/np.mean(r)) if len(r)>5 else np.nan for r in rrs_hr])
        hrv_calc = np.array([np.std(r) if len(r)>100 else np.nan for r in rrs])
        hrv_list.append(np.mean(hrv_calc))
        hr_list.append(np.mean(hr))
        hrv_prop_calc.append(np.sum(np.isnan(hrv_calc))/len(hrv_calc))
        hr_prop_calc.append(np.sum(np.isnan(hr))/len(hr))

        errors.append(noise)
        
    weartimes['errorrate'] = errors
    if len(weartimes[weartimes.Attached=='Yes'])>0:
        print('Error rate (while attached): ' + str(weartimes[weartimes.Attached=='Yes'].errorrate.mean().round(3)) +  ' non-physiological peaks per minute')
    weartimes_all = weartimes_all.append(weartimes)


