#Import packages
import numpy as np
from baseband import vdif
import astropy.units as u
import glob
from baseband.helpers import sequentialfile as sf
import pyfftw.interfaces.numpy_fft as fftw
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

np.seterr(divide='ignore', invalid='ignore')
from matplotlib import cm

#Single file to be searched
file = "/scratch/p/pen/hsiuhsil/daily_vdif_aro/B0531+21/81949485/0007605586.vdif"
#Constants
sample_rate = 400/1024*u.MHz
ftop = 800 * u.MHz - 400*np.arange(1024)/1024 * u.MHz
D = 4148.808 * u.s * u.MHz**2 * u.cm**3 / u.pc
fref = 800. * u.MHz
DM = 56.7546 * u.pc / u.cm**3
dt = 2.56e-6 * u.s

wrap_time = D*(1/(ftop[-1]**2)-1/(ftop[0]**2))*DM
wrap = int(np.ceil(D*(1/(ftop[-1]**2)-1/(ftop[0]**2))*DM/dt))
print(wrap)
dd_duration = 0.5*u.s
ntime = int((wrap_time+dd_duration)/dt)
nchan = 1024
npol = 2

#Opening single file and seeing data details
fh = vdif.open(file, 'rs', sample_rate = sample_rate)
print(fh.info())

#Creating a list of file to be synthesized.
filenames = sorted(glob.glob('/scratch/p/pen/hsiuhsil/daily_vdif_aro/B0531+21/81949485/*.vdif'))

#Synthesizing files and opening contiguous.
fraw = sf.open(filenames, 'rb')
fh2 = vdif.open(fraw, 'rs', sample_rate = sample_rate)

#matrix containing raw data
z = fh2.read(ntime).astype(np.complex64)

def coherent_dedispersion(z, DM, channel, axis=0):
    """Coherently dedisperse signal."""

    fcen = ftop[channel]

    f = fcen + np.fft.fftfreq(z.shape[axis], dt)
    dang = D * DM * u.cycle * f * (1./fref - 1./f)**2
    with u.set_enabled_equivalencies(u.dimensionless_angles()):
        dd_coh = np.exp(dang * 1j).conj().astype(np.complex64).value
    if z.ndim > 1:
        ind = [np.newaxis] * z.ndim
        ind[axis] = slice(None)
    if z.ndim > 1: 
        dd_coh = dd_coh[ind]
    z = fftw.fft(z, axis=axis)
    z = fftw.ifft(z * dd_coh, axis=axis)
    return z

#Applying coherent de-dispersion
for channel in range(nchan):
    if channel%100==0:
        print('channel:',channel)
    z[..., channel] = coherent_dedispersion(z[..., channel], DM, channel)

z_cohdd = z[:-wrap].transpose(2,1,0)
#np.save('/home/p/pen/pthulasi/cita_surp_2020/cta200_project/z_cohdd.npy', z_cohdd)
print(z_cohdd.shape, "saved, done")

