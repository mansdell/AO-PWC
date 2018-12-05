### IMPORT PACKAGES
import sys, os, pdb, glob, time
import numpy as np

### LOAD SIMULATED PHASE SCREENS
data = np.load('/data_sata1/ao_phase/phase_screens/phase_screens_part1.npy')

### PRINT SHAPE TO CHECK
data.shape
# (50000, 100, 19, 19)

### DEFINE VALUES FOR TRANSORMING DATA
wavelength_nm = 658
photons_per_elem = 4800 
snr_per_elem = photons_per_elem / np.sqrt(photons_per_elem)
measurement_error = (1 / snr_per_elem) * wavelength_nm / (2*np.pi)
print("SNR per element: " + str(round(snr_per_elem, 2)))
print("Measurement error: " + str(round(measurement_error, 2)))
# SNR per element: 69.28
# Measurement error: 1.51

### CHANGE RADIANS TO NANOMETRS
data = data * wavelength_nm / (2 * np.pi)

### ADD SHOT NOISE
data = data + np.random.normal(0, measurement_error, np.shape(data))

### CALCULATE MEAN AND STANDARD DEVIATION ACROSS ENTIRE DATASET
data_mean = np.nanmean(data)
data_stdev = np.nanstd(data)
print("\nmean of training dataset = {0:0.2f} nm".format(data_mean))
print("stdev of training dataset = {0:0.2f} nm".format(data_stdev))
# mean of training dataset = -9.48 nm
# stdev of training dataset = 2163.02 nm

### NORMALIZE DATA
data -= data_mean
data /= data_stdev
print("\nmean of training dataset after normalization = {0:0.2f} nm".format(np.nanmean(data)))
print("stdev of training dataset after normalization = {0:0.2f} nm".format(np.nanstd(data)))
# mean of training dataset after normalization = 0.0
# stdev of training dataset after normalization = 1.0

### LOOP THROUGH SEGMENTS AND OUTPUT TO INDIVIDUAL FILES
outdir = '/data/ao_phase/phase_screens_part1/'
for idx in range(np.shape(data)[0]):
    
    ### GRAB SEGMENT; TRANSFORM INTO NUMPY ARRAY
    dataset = np.array(data[idx,:,:,:])
    
    ### OUTPUT FRAME
    np.save(os.path.join(outdir, 'phase_screens_' + str(idx).zfill(5)), dataset)
