import numpy as np
import matplotlib.pyplot as plt
import scipy.signal, os

def read_IR(file):
	data = np.genfromtxt(file, skip_header=2, delimiter=',')[:,:-1]  #dont send last row cause itis empty
	return data

def get_peaks(transmittance, prominence=0):
	peak, _ = scipy.signal.find_peaks(transmittance, prominence=prominence, height=0)
	#get peak_height
	peak_height_0 = scipy.signal.peak_widths(transmittance, peak, rel_height=0)[1]
	peak_height_1 = scipy.signal.peak_widths(transmittance, peak, rel_height=1)[1]
	return peak, peak_height_0 - peak_height_1

def get_adjacent_peaks(y, xs):
	#gets values in xs that are left and right of value y
	xs = np.sort(xs)
	
	for i, x in enumerate(xs):
		next_x = xs[i+1]
		if x <= y <= next_x:
			return x, next_x

def fit_FSR(peakx, FSRinit=100, maxiter=1000, eps=1e-8):
	#probably not mathematically sound
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



def fit_exp_decay(x, y, maxiter=100_000, eps=1e-12, plot_errors=True, show_fit=True, plots_dir='', title='Predicted and reference rate', outfile=None, index=0):
	#x is usually time, y is usually tracked wavelength
	#fitting to [A]_0 exp(-kt) + B
	#initial values
	k = 2/x.max()			#height of 0.1 + B at time x.max()/2
	A0 = (y.max()-y.min()) 	
	B = y.min() 			

	f = lambda x, A0, k, B: A0 * np.exp(-k*x) + B
	error = lambda x, A0, k, B: np.sum((y-f(x, A0, k, B))**2)

	# plt.plot(x, y)
	# cmap = plt.get_cmap('hot')
	errors = []
	for i in range(maxiter):
		errors.append(error(x, A0, k, B))
		# if errors[-1] < eps: break #check for completion of minimization
		if i > 10 and all([(e - errors[-1]) < eps for e in errors[-8:]]): break

		#calculate derivatives
		expdec = np.exp(-k*x)
		base = A0*expdec + B - y
		dA0 = np.sum(2*expdec			* base)
		dk  = np.sum(2*x*A0*expdec*-1	* base)
		dB  = np.sum(2			 		* base)

		#change parameters
		A0 = A0 - dA0 * 0.01
		k = k - dk * 0.01
		B = B - dB * 0.01

	# if i < maxiter:
	# 	print(f'\tExponential decay fit converged in {len(errors)}/{maxiter} iterations.', file=outfile)
	# else:
	# 	print(f'\tExponential decay fit not converged. Will take more than {maxiter} iterations, or is not possible. ', file=outfile)

	if plot_errors: 
		plt.figure()
		plt.plot(range(len(errors)), errors)
		plt.title(f'Error during fitting {index}')
		plt.xlabel('Iteration')
		plt.ylabel('Error')
		plt.savefig(f'{plots_dir}/fitting_errors_{index}.jpg')

	if show_fit: 
		plt.figure()
		plt.plot(x, f(x, A0, k, B), label='Predicted decay')
		plt.plot(x, y, label='Reference data')
		plt.title(f'{title} {index}\nA0={A0:.3f}, k={k:.3f}, B={B:.3f}')
		plt.xlabel('Time (s)')
		plt.ylabel('Transmittance')
		plt.hlines(B + A0/2, x.min(), np.log(2)/k, colors=['black'], linewidths=[.75])
		plt.vlines(np.log(2)/k, y.min(), B + A0/2, colors=['black'], linewidths=[.75])
		plt.scatter(np.log(2)/k, B + A0/2, c='black')
		plt.text(np.log(2.2)/k, B + A0/1.85, r'$t_{1/2} = $' + f'{np.log(2)/k:.2f} s')
		plt.legend()
		plt.savefig(f'{plots_dir}/exp_model_fit_{index}.jpg')

	results = {'A0':A0, 'k':k, 'B':B, 'iterations':i, 'error':errors[-1]}
	return results

def get_tracking_transmittances(wavelength, wavelenghts, transmittances):
	low = wavelength[0]
	high = wavelength[1]

	prev_wl = wavelenghts.max()
	for i, wl in enumerate(wavelenghts):
		if prev_wl <= low  <= wl or prev_wl >= low  >= wl:
			low_i  = i-1
		if prev_wl <= high <= wl or prev_wl >= high >= wl:
			high_i = i-1

		prev_wl = wl

	if low_i > high_i:
		low_i, high_i = high_i, low_i
	return np.mean(transmittances[:,low_i:high_i], axis=1)


#constants
lightspeed 	= 2.99792458e10 #cm/s
planck 		= 4.135667696e-15 #eV s



if __name__ == '__main__':
	## important settings
	file = 'data/20211108 emptry cavity/repeated.csv'
	name = 'empty_cavity_test'
	convert_absorbance_to_transmittance = False
	selected_spectra = [] #indices of spectra you want to consider. leave empty for all
	polariton_wns = [] #wavenumbers where you expect polaritons cm^-1

	# kinetics settings
	tracking_wavelengths = [(6750, 6882), (2850, 2925)] #list of tuples where each tuple t has low and high wl
	time_delay = 12 #in seconds, time between measurements

	#fitting settings
	fit_maxiter = 100_000
	fit_eps = 1e-12

	## less important settings
	cmap = plt.get_cmap('jet')
	plot_fringes = False
	plot_fringes_for_one = True
	show_tracking_wavelengths = True



	## rest of program
	# open(f'{results}/{name}.log', 'w').close()
	if not os.path.exists(f'results/{name}'):
		os.mkdir(f'results/{name}')
	logfile = open(f'results/{name}/python.log', 'w')
	plots_dir = f'results/{name}/plots'
	if not os.path.exists(plots_dir):
		os.mkdir(plots_dir)

	print('=== GENERAL', file=logfile)

	data = read_IR(file)
	wavelenghts = data[:,0]
	wavelenght_stepsize = -np.diff(wavelenghts).mean()
	transmittances = data[:,1:]

	if convert_absorbance_to_transmittance:
		transmittances = 10**-transmittances

	transmittances = [transmittances[:,i] for i in range(transmittances.shape[1]) if i in selected_spectra or len(selected_spectra) == 0]
	transmittances = np.asarray(transmittances)
	min_abs = min([a.min() for a in transmittances])
	max_abs = max([a.max() for a in transmittances])

	N_spectra = len(transmittances)

	print(f'Filename: {file}', file=logfile)
	print(f'Loaded {N_spectra} spectra!', file=logfile)
	print(f'v_range  =  [{wavelenghts[0]}, {wavelenghts[-1]}] cm^-1', file=logfile)
	print(f'deltav   =  {wavelenght_stepsize:.4f} cm^-1', file=logfile)

	plt.figure(figsize=(16,9))
	plt.title('Spectra')
	plt.xlabel('v (cm^-1)')
	plt.ylabel('Transmittance')
	for i, a in enumerate(transmittances):
		plt.plot(wavelenghts, a , c=cmap(i/N_spectra))

	tracking_wavelengths_colors = cmap(np.linspace(0, 1, len(tracking_wavelengths)))
	if show_tracking_wavelengths:
		# plt.vlines([twl[0] for twl in tracking_wavelengths], min_abs, max_abs, colors=tracking_wavelengths_colors)
		# plt.vlines([twl[1] for twl in tracking_wavelengths], min_abs, max_abs, colors=tracking_wavelengths_colors)
		for twl, c in zip(tracking_wavelengths, tracking_wavelengths_colors):
			plt.fill_betweenx(np.array([min_abs, max_abs]), twl[0], twl[1], color=c, alpha=0.25)

	#calculations
	#Rabi splitting
	peaks = []
	peaks_for_fsr = []
	peak_heights = []
	peak_heights_for_fsr = []
	for a in transmittances:
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
	print('\n=== SPECTRA', file=logfile)
	for a, p, pfsr, ph, phfsr in zip(transmittances, peaks, peaks_for_fsr, peak_heights, peak_heights_for_fsr):
		print(f'Spectrum {i}:', file=logfile)

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
		print(f'\tFSR = {FSR:.2f} cm^-1, offset = {offset:.2f} cm^-1', file=logfile)

		#get Q-factor: Q = FSR/FWHM
		peak_widths_FSR = scipy.signal.peak_widths(a, fit_psfr, rel_height=0.5)
		FWHM = [pw*wavelenght_stepsize for pw in peak_widths_FSR][0].mean()

		Q = FSR/FWHM
		print(f'\tQuality = {Q:.3f}', file=logfile)


		if plot_fringes or (plot_fringes_for_one and i==0):
			plt.vlines([n*FSR + offset for n in range(-100, 100) if wavelenghts.min() < (n*FSR + offset) < wavelenghts.max()], min_abs, max_abs, linestyle='dashed', linewidths=0.75)
 
		#calculate Rabi splitting
		for j, polariton_wn in enumerate(polariton_wns):
			print(f'\tPolariton {j} @ {polariton_wn} cm^-1', file=logfile)
			l, h = get_adjacent_peaks(polariton_wn, wavelenghts[p])
			plt.vlines((l,h), min_abs, max_abs, colors='red', linewidths=1)
			splitting = (h-l)*planck*lightspeed
			idx = np.where(np.logical_or(wavelenghts == l, wavelenghts == h))[0]
			peak_props = scipy.signal.peak_widths(a, idx)
			peak_widths = [pw*wavelenght_stepsize for pw in peak_props][0]

			print(f'\t\tP+ @ {int(h)} cm^-1', file=logfile)
			print(f'\t\tP- @ {int(l)} cm^-1', file=logfile)
			print(f'\t\tRabi splitting = {splitting:.5f} eV', file=logfile)
			# print(f'\t\theight(P+)/height(P-) = {peak_widths[1]/peak_widths[0]}', file=logfile)

		# print('\n', file=logfile)
		i += 1

	plt.savefig(f'{plots_dir}/spectra_main.jpg')

	##kinetics:
	if len(tracking_wavelengths) > 0:

		print('\n=== KINETICS', file=logfile)
		print(f'{len(tracking_wavelengths)} wavelengths being tracked:', file=logfile)
		for twl in tracking_wavelengths:
			print(f'\tAverage over [{twl[0]}, {twl[1]}] cm^-1', file=logfile)

		tracking_transmittances = []
		for tracking_wavelength in tracking_wavelengths:
			tt = get_tracking_transmittances(tracking_wavelength, wavelenghts, transmittances)
			tracking_transmittances.append(tt)

		#plot the tracked wls
		plt.figure()
		plt.title('Tracked wavelengths')
		plt.xlabel('t (s)')
		plt.ylabel('Transmittance')
		for twl, tabs, c in zip(tracking_wavelengths, tracking_transmittances, tracking_wavelengths_colors):
			low = twl[0]
			high = twl[1]
			plt.plot(np.arange(N_spectra)*time_delay, tabs, label=r'$v_{avg}$' + f' = [{low}, {high}] cm^-1', c=c)
		plt.legend()
		plt.savefig(f'{plots_dir}/tracked_wavelengths.jpg')

		#plot the normalised tracked wls
		plt.figure()
		plt.title('Tracked wavelengths normalised')
		plt.xlabel('t (s)')
		plt.ylabel('Transmittance (normalised scale)')
		for twl, tabs, c in zip(tracking_wavelengths, tracking_transmittances, tracking_wavelengths_colors):
			low = twl[0]
			high = twl[1]
			tabsn = tabs - tabs.min()
			tabsn = tabsn / tabsn.max()
			plt.plot(np.arange(N_spectra)*time_delay, tabsn, label=r'$v_{avg}$' + f' = [{low}, {high}] cm^-1', c=c)
		plt.legend()
		plt.savefig(f'{plots_dir}/tracked_wavelengths_normalised.jpg')


		#fit exp decay
		print(f'\nFitting {len(tracking_wavelengths)} curves using exponential decay:', file=logfile)
		print(f'\tSettings', file=logfile)
		print(f'\t\tmaxiter = {fit_maxiter}', file=logfile)
		print(f'\t\teps     = {fit_eps}', file=logfile)
		for i, tabs in enumerate(tracking_transmittances):
			fit_results = fit_exp_decay(np.arange(N_spectra)*time_delay, tabs, index=i, plots_dir=plots_dir, outfile=logfile, maxiter=fit_maxiter, eps=fit_eps)

			print(f'\tFit {i}', file=logfile)
			if fit_results['iterations'] <= fit_maxiter:
				print(f'\t\tModel converged in {fit_results["iterations"]} iterations', file=logfile)
			else:
				print(f'\t\tModel did not converge in {fit_results["iterations"]} iterations, try different settings or another model', file=logfile)

			k = fit_results["k"]
			print(f'\t\tA0      = {fit_results["A0"]}', file=logfile)
			print(f'\t\tk       = {k} s^-1', file=logfile)
			print(f'\t\tB       = {fit_results["B"]}', file=logfile)
			print(f'\t\tError   = {fit_results["error"]}', file=logfile)
			print(f'\t\tMean Lifetime 	= {1/k:.2f} s', file=logfile)
			print(f'\t\tHalf-life		= {np.log(2)/k:.2f} s', file=logfile)

	print('\n=== END', file=logfile)

	print(f'Plots saved to {plots_dir}', file=logfile)
	# plt.show()