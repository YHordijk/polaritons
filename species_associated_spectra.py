import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors
import scipy.signal
from pkg.exp_decay_fitter import fit_exp_decay, fit_biexp_decay, fit_exp_lin
import os


def read_UV(file):
	data = np.genfromtxt(file, skip_header=1, delimiter=',')[:,:-1]
	return data


def get_TRS(file):
	data = read_UV(file)
	spectray = data[:,1:]
	spectrax = data[:,0]

	# spectray = spectray[np.logical_and(400 < spectrax, spectrax < 1000)][:,:60]
	# spectrax = spectrax[np.logical_and(400 < spectrax, spectrax < 1000)]

	Nspectra = spectray.shape[1]
	extent =  spectrax.min(), spectrax.max(), 0, Nspectra*10

	plt.ylabel('Time (s)')
	plt.xlabel('$\lambda$ (nm)')
	plt.imshow(spectray.T, extent=extent, aspect='auto', origin='lower')
	plt.show()

	return spectrax, spectray.T


def get_SAS(x, y):
	#f(lambda, t) = sum(f_i(lambda, t))
	...



if __name__ == '__main__':
	TRSx, TRSy = get_TRS('data/20211126_UVVIS_SP_cyclohexanone_5um_coupled/data.csv')