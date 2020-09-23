import numpy as np
import matplotlib.pyplot as plt
import sys, os

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# constants
tsamp = 0.001048576
fref = 1405.0

# parameters for run
dm_min = 0.0
dm_max = 2000.0
dm_tol = 1.35

boxcar_min = 1.0
boxcar_max = 32.0
linear = True

nfrbs = 10000

# get dm trials
os.system('/home/dsa/dsa110-mbheimdall/bin/generate_dmlist -f0 1530.0 -df -0.244140625 -nchan 1024 -dt 0.001048576 -dm '+str(dm_min)+' '+str(dm_max)+' -dm_tol '+str(dm_tol)+' > output.dms')
dms = np.loadtxt('output.dms')
print(len(dms))

# get boxcars
if linear:
    boxcars = np.arange(boxcar_min,boxcar_max+1)
else:
    boxcars = []
    mb = boxcar_min
    while mb <= boxcar_max:
        boxcars.append(mb)
        mb *= 2.
    boxcars = np.asarray(boxcars)

# generate FRBs
frb_dms = np.random.uniform(size=nfrbs)*(dm_max-dm_min)+dm_min
frb_widths = np.random.uniform(size=nfrbs)*(boxcar_max-boxcar_min)+boxcar_min

# smear them
snrs = np.zeros(nfrbs)+1.
for i in range(nfrbs):

    dm_nearest = find_nearest(dms,frb_dms[i])
    wid_obs = np.sqrt((np.abs(0.00415*np.abs(dm_nearest-frb_dms[i])*(1./1.28**2-1./1.53**2.))**2.) + (tsamp*frb_widths[i])**2.)
    w_nearest = find_nearest(boxcars,wid_obs/tsamp)
    
    snrs[i] = np.sqrt(frb_widths[i]*tsamp/wid_obs)
    if w_nearest<wid_obs/tsamp:
        snrs[i] *= np.sqrt(w_nearest*tsamp/wid_obs)
    else:
        snrs[i] *= np.sqrt(wid_obs/(w_nearest*tsamp))
    

print('5th perc S/N loss: ',np.percentile(snrs,5.))
# plot them
#plt.figure(figsize=(10.,10.))
#plt.xlabel('DM')
#plt.ylabel('SNR')
#plt.plot(frb_dms,snrs,'.')
#plt.show()


    
    
