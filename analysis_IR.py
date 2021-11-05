import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

def read_IR(file):
	data = np.genfromtxt(file, skip_header=2, delimiter=',')[:,:-1]  #dont send last row cause itis empty
	return data

def get_peaks(absorbance, prominence=0):
	peak, _ = scipy.signal.find_peaks(absorbance, prominence=prominence, height=0)
	#get peak_height
	peak_height_0 = scipy.signal.peak_widths(absorbance, peak, rel_height=0)[1]
	peak_height_1 = scipy.signal.peak_widths(absorbance, peak, rel_height=1)[1]
	return peak, peak_height_0 - peak_height_1

def get_adjacent_peaks(y, xs):
	#gets values in xs that are left and right of value y
	xs = np.sort(xs)
	
	for i, x in enumerate(xs):
		next_x = xs[i+1]
		if x <= y <= next_x:
			return x, next_x

def fit_FSR(peakx, FSRinit=100, maxiter=1000, eps=1e-8):
	def error(FSR, ns):
		kmin = ns[np.argmin(np.abs(ns*FSR - peakx), axis=0)]
		return np.sum((kmin*FSR - peakx)**2)

	def deriv(FSR, ns):
		kmin = ns[np.argmin(np.abs(ns*FSR - peakx), axis=0)]
		return np.sum(2*kmin)

	FSR = FSRinit
	errors = []
	for i in range(maxiter):
		ns = np.arange(peakx.min()//FSR, peakx.max()//FSR+2).reshape(-1,1)
		errors.append(error(FSR, ns))
		if i > 10 and all([abs(e - errors[-1]) < eps for e in errors[-8:]]): break #stop condition
		FSR = FSR - 0.00001 * deriv(FSR, ns)
	return FSR

#constants
lightspeed 	= 2.99792458e10 #cm/s
planck 		= 4.135667696e-15 #eV s



if __name__ == '__main__':
	file = 'data/20211102 IRs acetone cavity/all_Irs.csv'
	selected_spectra = [3]
	polariton_wns = [1725, 3005] #wavenumbers where you expect polaritons cm^-1

	data = read_IR(file)
	wavelenghts = data[:,0]
	wavelenght_stepsize = -np.diff(wavelenghts).mean()
	absorbances = data[:,1:]

	absorbances = [absorbances[:,i] for i in range(absorbances.shape[1]) if i in selected_spectra]

	min_abs = min([a.min() for a in absorbances])
	max_abs = max([a.max() for a in absorbances])

	cmap = plt.get_cmap('jet')
	N_spectra = len(absorbances)

	print(f'Loaded {N_spectra} spectra!')
	print(f'v ⊂ [{wavelenghts[0]}, {wavelenghts[-1]}] cm^-1')
	print(f'Δv = {wavelenght_stepsize:.4f} cm^-1')

	plt.figure()
	plt.title('Spectra')
	plt.xlabel('v (cm^-1)')
	plt.ylabel('Absorbance')
	for i, a in enumerate(absorbances):
		plt.plot(wavelenghts, a , c=cmap(i/N_spectra))

	#calculations
	#Rabi splitting
	peaks = []
	peaks_for_fsr = []
	peak_heights = []
	peak_heights_for_fsr = []
	for a in absorbances:
		peak, ph = get_peaks(a, prominence=0.005)
		# plt.scatter(wavelenghts[peak], a[peak])
		peaks.append(peak)
		peak_heights.append(ph)

		peak, ph = get_peaks(a, prominence=0.005)
		# plt.scatter(wavelenghts[peak], a[peak])
		peak_heights_for_fsr.append(ph)
		peaks_for_fsr.append(peak)


	#get differences in spectra for calculation of FSR
	FSRs = []
	i = 0
	for a, p, pfsr, ph, phfsr in zip(absorbances, peaks, peaks_for_fsr, peak_heights, peak_heights_for_fsr):
		print(f'Spectrum {0}:')

		#fit FSR
		#get wavelenghts used to fit
		# w = wavelenghts[pfsr[-7:-1]]
		fit_psfr = pfsr[wavelenghts[pfsr] > 4000]
		w = wavelenghts[fit_psfr]

		diff = -np.diff(w)
		diff = diff[np.abs(diff-diff.mean()) < 20] #removelarge errors
		FSRinit = np.mean(diff)
		offset = w[-1]%FSRinit

		#make better
		# FSR = fit_FSR(wavelenghts[pfsr], FSRinit=FSRinit) #DOES NOT WORK
		FSR = FSRinit
		print(f'\tFSR = {FSR:.2f} cm^-1, offset={offset:.2f} cm^-1')

		#get Q-factor: Q = FSR/FWHM
		peak_widths_FSR = scipy.signal.peak_widths(a, fit_psfr, rel_height=0.5)
		FWHM = [pw*wavelenght_stepsize for pw in peak_widths_FSR][0].mean()

		Q = FSR/FWHM
		print(f'\tQuality = {Q:.3f}')

		
		plt.vlines([n*FSR + offset for n in range(-100, 100) if wavelenghts.min() < (n*FSR + offset) < wavelenghts.max()], min_abs, max_abs, linestyle='dashed', linewidths=0.75)
 
		#calculate Rabi splitting
		for polariton_wn in polariton_wns:
			print(f'\tPolariton @ {polariton_wn} cm^-1')
			l, h = get_adjacent_peaks(polariton_wn, wavelenghts[p])
			plt.vlines((l,h), min_abs, max_abs, colors='red', linewidths=1)
			splitting = (h-l)*planck*lightspeed
			idx = np.where(np.logical_or(wavelenghts == l, wavelenghts == h))[0]
			peak_props = scipy.signal.peak_widths(a, idx)
			peak_widths = [pw*wavelenght_stepsize for pw in peak_props][0]

			print(f'\t\tP+ @ {int(h)} cm^-1')
			print(f'\t\tP- @ {int(l)} cm^-1')
			print(f'\t\tRabi splitting = {splitting:.5f} eV')
			print(f'\t\theight(P+)/height(P-) = {peak_widths[1]/peak_widths[0]}')

		print('\n')
		i += 1

	plt.show()