import numpy as np
import matplotlib.pyplot as plt
import scipy.signal, os
from pkg.lorentz_fitter import fit_lorentzian
from pkg.exp_decay_fitter import fit_exp_decay
from pkg.peak_finding import *

def read_csv(file):
	with open(file, 'r') as f:
		spectrum_types = f.readlines()[1].lower().strip().strip(',').split(',')[1:]
	data = np.genfromtxt(file, skip_header=2, delimiter=',')[:,:-1]  #dont send last row cause itis empty
	return data, spectrum_types


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


def get_tracking_spectra(wavelength, spectrax, spectra):
	low = wavelength[0]
	high = wavelength[1]

	prev_wl = spectrax.max()
	for i, wl in enumerate(spectrax):
		if prev_wl <= low  <= wl or prev_wl >= low  >= wl:
			low_i  = i-1
		if prev_wl <= high <= wl or prev_wl >= high >= wl:
			high_i = i-1

		prev_wl = wl

	if low_i > high_i:
		low_i, high_i = high_i, low_i
	return np.mean(spectra[:,low_i:high_i], axis=1)


#constants
lightspeed 	= 2.99792458e10 #cm/s
planck 		= 4.135667696e-15 #eV s


def main(**settings):
	#get default settings
	name = settings.get('name', '')
	file = settings.get('file', '')
	convert_absorbance_to_transmittance = settings.get('convert_absorbance_to_transmittance', False)
	spectrum_type = settings.get('spectrum_type', 'transmittance')
	selected_spectra = settings.get('selected_spectra', [])
	polariton_wns = settings.get('polariton_wns', [])
	refractive_index = settings.get('refractive_index', 1.3592) #default to acetone

	#kinetics settings
	tracking_spectrax = settings.get('tracking_spectrax', [])
	time_delay = settings.get('time_delay', 1)
	fit_maxiter = settings.get('fit_maxiter', 100_000)
	fit_eps = settings.get('fit_eps', 1e-12)

	#less important settings
	cmap = settings.get('cmap', plt.get_cmap('jet'))

	#plotting settings
	show_tracking_spectrax = settings.get('show_tracking_spectrax', True)
	plot_polaritons = settings.get('plot_polaritons', True)
	plot_peaks = settings.get('plot_peaks', False)
	plot_fringes = settings.get('plot_fringes', False)
	plot_fringes_for_one = settings.get('plot_fringes_for_one', True)
	overlay_spectrum = settings.get('overlay_spectrum', True)
	overlay_spectrum_type = settings.get('overlay_spectrum_type', 'absorbance')
	overlay_spectrum_path = settings.get('overlay_spectrum_path', 'data/aceton_IR_spectrum_2.csv')
	overlay_spectrum_inverted = settings.get('overlay_spectrum_inverted', True)
	show_plots = settings.get('show_plots', False)



	## rest of program
	if not os.path.exists(f'results/{name}'):
		os.mkdir(f'results/{name}')
	logfile = open(f'results/{name}/python.log', 'w')
	spectrum_data_path = f'results/{name}/spectrum_data.csv'
	plots_dir = f'results/{name}/plots'
	if not os.path.exists(plots_dir):
		os.mkdir(plots_dir)


	print('=== GENERAL', file=logfile)

	data, spectrum_types = read_csv(file)

	if overlay_spectrum:
		extra_spectrum, extra_spectrum_type = read_csv(overlay_spectrum_path)
		extra_spectrum_type = extra_spectrum_type[0]
		overlay_x, overlay_y = extra_spectrum[:,0], extra_spectrum[:,1]
		
		if not overlay_spectrum_type == extra_spectrum_type:
			if overlay_spectrum_type == 'transmittance':
				overlay_y = 10**-overlay_y
			elif overlay_spectrum_type == 'absorbance':
				overlay_y = -np.log10(overlay_y)
				# print(overlay_y)

		if overlay_spectrum_inverted: overlay_y = -overlay_y

	spectrax = data[:,0]
	wavelenght_stepsize = -np.diff(spectrax).mean()
	spectra = data[:,1:]

	#convert spctra
	converted_spectra = []
	for st, spectrum in zip(spectrum_types, data.T):
		if not st == spectrum_type:
			if spectrum_type == 'transmittance':
				s = 10**-spectrum
			else:
				s = -np.log(spectrum)
		else:
			s = spectrum

		converted_spectra.append(s)

	spectra = [spectra[:,i] for i in range(spectra.shape[1]) if i in selected_spectra or len(selected_spectra) == 0]
	spectra = np.asarray(spectra)
	min_abs = min([a.min() for a in spectra])
	max_abs = max([a.max() for a in spectra])

	N_spectra = len(spectra)

	print(f'Filename: {file}', file=logfile)
	print(f'Loaded {N_spectra} spectra!', file=logfile)
	print(f'v_range  =  [{spectrax[0]}, {spectrax[-1]}] cm^-1', file=logfile)
	print(f'deltav   =  {wavelenght_stepsize:.4f} cm^-1', file=logfile)

	plt.figure(figsize=(16,9))
	plt.title('Spectra')
	plt.xlabel('v (cm^-1)')
	plt.ylabel('Transmittance')
	# for i, a in enumerate(spectra):
	# 	plt.plot(spectrax, a , c=cmap(i/N_spectra))

	if overlay_spectrum:
		#normalize aceton_y
		overlay_y = (overlay_y - overlay_y.min())/(overlay_y.max() - overlay_y.min()) * max_abs
		plt.plot(overlay_x, overlay_y, label=f'aceton IR {overlay_spectrum_type} spectrum' + ', inverted'*overlay_spectrum_inverted, linewidth=0.5, color='black')

	tracking_spectrax_colors = cmap(np.linspace(0, 1, len(tracking_spectrax)))
	if show_tracking_spectrax:
		for twl, c in zip(tracking_spectrax, tracking_spectrax_colors):
			plt.fill_betweenx(np.array([min_abs, max_abs]), twl[0], twl[1], color=c, alpha=0.25)

	# #calculations
	# #Rabi splitting
	# peaks = []
	# peaks_for_fsr = []
	# peak_heights = []
	# peak_heights_for_fsr = []

	# FWHMs_fit = []
	# for i, a in enumerate(spectra):
	# 	peak_res = get_peaks(spectrax, a, prominence=0.002)
	# 	peak_heights.append(peak_res['peaky'])
	# 	FWHMs_fit.append(peak_res['FWHM'])
	# 	if plot_peaks: plt.scatter(spectrax[peak], a[peak], color=cmap(i/N_spectra))
	# 	peaks.append(peak)
	# 	peak_heights.append(ph)

	# 	peak, ph = get_peaks(spectrax, a, prominence=0.05, debug=False)
	# 	# plt.scatter(spectrax[peak], a[peak])
	# 	peak_heights_for_fsr.append(ph)
	# 	peaks_for_fsr.append(peak)

	# #get differences in spectra for calculation of FSR
	# FSRs = []
	# FWHMs = []

	# i = 0
	print('\n=== SPECTRA', file=logfile)
	for i, spectrum in enumerate(spectra):
		color = cmap(i/N_spectra)
		print(f'Spectrum {i}:', file=logfile)

		plt.plot(spectrax, spectrum, color=color)

		#all peaks
		peak_res = get_peaks(spectrax, spectrum, prominence=0.003 )
		peak_res_fsr = get_peaks(spectrax, spectrum, prominence=0.05)

		if plot_peaks:
			plt.scatter(peak_res['peakx'], peak_res['peaky'], color=color)

		FSR_res = get_FSR(peak_res_fsr['peakx'], 4000)
		FSR = FSR_res['FSR']
		FSR_offset = FSR_res['offset']

		print(f'\tFSR            = {FSR:.2f} cm^-1', file=logfile)
		print(f'\tFSR_offset     = {FSR_offset:.2f} cm^-1', file=logfile)

		#cavity_spacing:
		spacing = 10**4/(2*refractive_index*FSR)
		print(f'\tCavity spacing = {spacing:.4f} um', file=logfile)

		#get Q-factor: Q = FSR/FWHM
		FWHM = peak_res_fsr['FWHM'].mean()

		Q = FSR/FWHM
		print(f'\tFWHM           = {FWHM:.3f} cm^-1', file=logfile)
		print(f'\tQuality        = {Q:.3f}', file=logfile)


		if plot_fringes or (plot_fringes_for_one and i==0):
			plt.vlines([n*FSR + FSR_offset for n in range(-100, 100) if spectrax.min() < (n*FSR + FSR_offset) < spectrax.max()], min_abs, max_abs, linestyle='dashed', linewidths=0.75)


		#calculate Rabi splitting
		for j, polariton_wn in enumerate(polariton_wns):
			
			adjpeak = get_adjacent_peaks(polariton_wn, peak_res['peakx'])
			if adjpeak is None:
				continue
			l, h = adjpeak

			print(f'\tPolariton {j} @ {polariton_wn} cm^-1', file=logfile)
			if plot_polaritons:
				plt.vlines((l,h), min_abs, max_abs, colors='red', linewidths=1)
			splitting = (h-l)*planck*lightspeed
			# idx = np.where(np.logical_or(spectrax == l, spectrax == h))[0]
			# peak_props = scipy.signal.peak_widths(a, idx)
			# peak_widths = [pw*wavelenght_stepsize for pw in peak_props][0]

			print(f'\t\tP+ @ {int(h)} cm^-1, dv = {h-polariton_wn:.1f} cm^-1', file=logfile)
			print(f'\t\tP- @ {int(l)} cm^-1, dv = {polariton_wn-l:.1f} cm^-1', file=logfile)
			print(f'\t\tRabi splitting = {splitting:.5f} eV', file=logfile)

		# print('\n', file=logfile)
		i += 1
	plt.tight_layout()
	plt.legend()
	plt.savefig(f'{plots_dir}/spectra_main.jpg')

	##kinetics:
	if len(tracking_spectrax) > 0:

		print('\n=== KINETICS', file=logfile)
		print(f'{len(tracking_spectrax)} spectrax being tracked:', file=logfile)
		for twl in tracking_spectrax:
			print(f'\tAverage over [{twl[0]}, {twl[1]}] cm^-1', file=logfile)

		tracking_spectra = []
		for tracking_wavelength in tracking_spectrax:
			tt = get_tracking_spectra(tracking_wavelength, spectrax, spectra)
			tracking_spectra.append(tt)

		#plot the tracked wls
		plt.figure()
		plt.title('Tracked spectrax')
		plt.xlabel('t (s)')
		plt.ylabel('Transmittance')
		for twl, tabs, c in zip(tracking_spectrax, tracking_spectra, tracking_spectrax_colors):
			low = twl[0]
			high = twl[1]
			plt.plot(np.arange(N_spectra)*time_delay, tabs, label=r'$v_{avg}$' + f' over [{low}, {high}] cm^-1', c=c)
		plt.legend()
		plt.tight_layout()
		plt.savefig(f'{plots_dir}/tracked_spectrax.jpg')

		#plot the normalised tracked wls
		plt.figure()
		plt.title('Tracked spectrax normalised')
		plt.xlabel('t (s)')
		plt.ylabel('Transmittance (normalised scale)')
		for twl, tabs, c in zip(tracking_spectrax, tracking_spectra, tracking_spectrax_colors):
			low = twl[0]
			high = twl[1]
			tabsn = tabs - tabs.min()
			tabsn = tabsn / tabsn.max()
			plt.plot(np.arange(N_spectra)*time_delay, tabsn, label=r'$v_{avg}$' + f' over [{low}, {high}] cm^-1', c=c)
		plt.legend()
		plt.tight_layout()
		plt.savefig(f'{plots_dir}/tracked_spectrax_normalised.jpg')


		#fit exp decay
		print(f'\nFitting {len(tracking_spectrax)} curves using exponential decay:', file=logfile)
		print(f'\tModel: A(t) = A0 * exp(-kt) + B', file=logfile)
		print(f'\tSettings', file=logfile)
		print(f'\t\tmaxiter         = {fit_maxiter}', file=logfile)
		print(f'\t\teps             = {fit_eps}', file=logfile)
		for i, tabs in enumerate(tracking_spectra):
			fit_results = fit_exp_decay(np.arange(N_spectra)*time_delay, tabs, index=i, plots_dir=plots_dir, outfile=logfile, maxiter=fit_maxiter, eps=fit_eps)
			print(f'\tResults fit {i}', file=logfile)
			k = fit_results["k"]
			if 'iterations' in fit_results:
				print(f'\t\tConvergence     = {fit_results["iterations"] <= fit_maxiter}', file=logfile)
				print(f'\t\tIterations      = {fit_results["iterations"]}', file=logfile)
			else:
				print(f'\t\tConvergence     = True', file=logfile)
			print(f'\t\tA0              = {fit_results["A0"]}', file=logfile)
			print(f'\t\tk               = {k} s^-1', file=logfile)
			print(f'\t\tB               = {fit_results["B"]}', file=logfile)
			print(f'\t\tError           = {fit_results["error"]}', file=logfile)
			print(f'\t\tMean Lifetime 	= {1/k:.2f} s', file=logfile)
			print(f'\t\tHalf-life		= {np.log(2)/k:.2f} s', file=logfile)

	print('\n=== END', file=logfile)

	print(f'Plots saved to {plots_dir}', file=logfile)

	# #write peak_data
	# with open(spectrum_data_path, 'w') as pd:
	# 	pd.write('spectrum, FSR, FWHM\n')
	# 	for i, fsr, fwhm in zip(range(N_spectra), FSRs, FWHMs):
	# 		pd.write(f'{i}, {fsr}, {fwhm}\n')
			
	print(f'Spectrum data written to {spectrum_data_path}', file=logfile)

	logfile.close()

	with open(f'results/{name}/python.log', 'r') as logfile:
		loglines = logfile.readlines()
		print(''.join(loglines))


	if show_plots: plt.show()



if __name__ == '__main__':
	settings = {
		'kinetics_1': {
			'file': 'data/20111111_IR_SP_kinetics_1/data.csv',
			'name': 'kinetics_1',
			'polariton_wns': [1716, 1220], #wavenumbers where you expect polaritons cm^-1
			'refractive_index': 1.3592, #aceton

			# kinetics settings
			'tracking_spectrax': [(6750, 6882), (2850, 2925)], #list of tuples where each tuple t has low and high wl
			'time_delay': 30, #in seconds, time between measurements

			## less important settings
			'plot_fringes_for_one': True,
			'show_tracking_spectrax': True,
			'plot_polaritons': True,
			'plot_peaks': True,
			},

		'kinetics_2': {
			'file': 'data/20111111_IR_SP_kinetics_2/data.csv',
			'name': 'kinetics_2',
			'polariton_wns': [1716, 1220], #wavenumbers where you expect polaritons cm^-1
			'refractive_index': 1.3592, #aceton,

			# kinetics settings
			'tracking_spectrax': [(6750, 6882), (2850, 2925)], #list of tuples where each tuple t has low and high wl
			'time_delay': 30, #in seconds, time between measurements

			## less important settings
			'plot_fringes_for_one': True,
			'show_tracking_spectrax': True,
			'plot_polaritons': True,
			'plot_peaks': True,
			},

		'cavity_tuning': {
			'file': 'data/20211112_cavity_tuning_for_acetone/data.csv',
			'name': 'cavity_tuning',
			'polariton_wns': [1716], #wavenumbers where you expect polaritons cm^-1
			# 'polariton_wns': [],
			'refractive_index': 1.3592, #aceton,

			## less important settings
			'plot_fringes_for_one': True,
			'plot_polaritons': True,
			'plot_peaks': True,
			},
	}

	# for n, s in settings.items():
	# 	print(n)
	# 	main(**s, show_plots=False)

	main(**settings['cavity_tuning'], show_plots=True)


