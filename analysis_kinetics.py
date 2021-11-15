import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from pkg.exp_decay_fitter import fit_exp_decay



def read_UV(file):
	data = np.genfromtxt(file, skip_header=1, delimiter=',')[:,:-1]
	return data



if __name__ == '__main__':
	### KINETICS

	# data = read_UV('data/20211102 kinetic nsp in cavity/kinetic_data_20sinterval_no_uv.csv')
	data = read_UV('data/20211102 kinetic nsp in cavity/kinetic_data_10sinterval.csv')
	time_delay = 20
	tracking_wavelengths = [224, 340, 500]
	tracking_wavelengths = [(520, 580)]
	tracking_wavelengths_k = 20 #wavelengths around tracking_wavelenghts to average around

	wavelenghts = data[:,0]
	absorbances = data[:,1:]
	cmap = plt.get_cmap('jet')
	N_spectra = absorbances.shape[1]

	print(f'Found {N_spectra} spectra! λ ⊂ [{wavelenghts[0]}, {wavelenghts[-1]}] nm')

	plt.figure()
	plt.title('Spectra')
	plt.xlabel('λ (nm)')
	plt.ylabel('Absorbance (a.u.)')
	for i, absorbance in enumerate(absorbances.T):
		plt.plot(wavelenghts, absorbance , c=cmap(i/N_spectra))

	
	def get_tracking_absorbance(wavelenght, k=5):
		if type(wavelenght) is tuple:
			low = wavelenght[0]
			high = wavelenght[1]
		else:
			low = wavelenght - k 
			high = wavelenght + k

		prev_wl = 0
		for i, wl in enumerate(wavelenghts):
			if prev_wl <= low and wl >= low:
				if k == 0:
					return absorbances[i-1] + (absorbances[i] - absorbances[i-1]) * (low-prev_wl)/(wl-prev_wl)
				else:
					low_i = i-1
			elif prev_wl <= high and wl >= high:
				high_i = i-1

			prev_wl = wl

		return np.mean(absorbances[low_i:high_i,:], axis=0)


	tracking_absorbances = []
	for twl in tracking_wavelengths:
		absorbs = get_tracking_absorbance(twl, tracking_wavelengths_k)
		tracking_absorbances.append(absorbs)

		if type(twl) is tuple: twlb = (twl[0]+twl[1])/2
		else: twlb = twl

		plt.scatter(np.ones_like(absorbs) * twlb, absorbs)

	plt.figure()
	plt.title('Tracked wavelenghts')
	plt.xlabel('t (s)')
	plt.ylabel('Absorbance (a.u.)')
	for twl, tabs in zip(tracking_wavelengths, tracking_absorbances):
		if type(twl) is tuple:
			low = twl[0]
			high = twl[1]
		else:
			low = twl - tracking_wavelengths_k
			high = twl + tracking_wavelengths_k
		
		plt.plot(np.arange(N_spectra)*time_delay, tabs, label=r'$\lambda_{avg}$' + f' = [{low}, {high}] nm')

	plt.legend()


	for twl, tabs in zip(tracking_wavelengths, tracking_absorbances):
		if type(twl) is tuple:
			low = twl[0]
			high = twl[1]
		else:
			low = twl - tracking_wavelengths_k
			high = twl + tracking_wavelengths_k

		fit_exp_decay(np.arange(N_spectra-6)*time_delay, tabs[6:], title=f'Predicted and reference rate (around $\lambda ∈ [{low}, {high}]$ nm)')


	plt.show()