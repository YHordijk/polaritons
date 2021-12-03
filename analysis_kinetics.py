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


def main(*args, **kwargs):
	## set paths

	#get settings
	name = kwargs['name']
	data = read_UV(kwargs['file'])
	time_delay = kwargs.get('time_delay', 1)
	tracking_spectrax = kwargs.get('tracking_spectrax', [])
	cmap = kwargs.get('cmap', plt.get_cmap('jet'))
	show_tracking_spectrax = kwargs.get('show_tracking_spectrax', True)
	spectra_blacklist = kwargs.get('spectra_blacklist', [])
	logfile = kwargs.get('logfile', None)
	result_dir = kwargs.get('result_dir', f'results/{name}')
	heatmap_maxt = kwargs.get('heatmap_maxt', None)


	if not os.path.exists(f'{result_dir}'):
		os.mkdir(f'{result_dir}')
	
	if logfile is None:
		logfile = open(f'{result_dir}/python.log', 'w')
	else:
		logfile = open(logfile, 'a')

	plots_dir = f'{result_dir}/plots'
	if not os.path.exists(plots_dir):
		os.mkdir(plots_dir)

	spectrax = data[:,0]
	spectra = data[:,1:]
	wavelenght_stepsize = abs(np.diff(spectrax).mean())

	min_abs = min([a.min() for a in spectra])
	max_abs = max([a.max() for a in spectra])
	
	N_spectra = spectra.shape[1]
	print('=== GENERAL', file=logfile)
	print(f'Loaded {N_spectra} spectra!', file=logfile)
	print(f'lambda_range  =  [{spectrax[0]}, {spectrax[-1]}] nm', file=logfile)
	print(f'stepsize      =  {wavelenght_stepsize:.4f} nm', file=logfile)

	plt.figure()
	plt.title('Spectra')
	plt.xlabel('λ (nm)')
	plt.ylabel('Absorbance (a.u.)')
	for i, absorbance in enumerate(spectra.T):
		plt.plot(spectrax, absorbance , c=cmap(i/N_spectra))


	tracking_spectrax_colors = cmap(np.linspace(0, 1, len(tracking_spectrax)))
	if show_tracking_spectrax:
		for twl, c in zip(tracking_spectrax, tracking_spectrax_colors):
			plt.fill_betweenx(np.array([min_abs, max_abs]), twl[0], twl[1], color=c, alpha=0.25)

	cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, N_spectra*time_delay), cmap=cmap))
	cbar.set_label(r'$\Delta t$')
	plt.savefig(f'{plots_dir}/spectra_main_kinetics.jpg')

	plt.figure()
	plt.title('Time resolved spectral heatmap')
	extent =  spectrax.min(), spectrax.max(), 0, N_spectra*time_delay

	plt.ylabel('Time (s)')
	plt.xlabel('$\lambda$ (nm)')
	plt.imshow(spectra.T, extent=extent, aspect='auto', origin='lower')
	cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(spectra.min(), spectra.max()), cmap=cmap))
	cbar.set_label('Absorbance (a.u.)')
	plt.savefig(f'{plots_dir}/spectra_heatmap_kinetics_all.jpg')

	for i, tx in enumerate(tracking_spectrax):
		plt.figure()
		plt.title(rf'Time resolved spectral heatmap between $\lambda \in$ [{tx[0]}, {tx[1]}] nm')

		maxt = heatmap_maxt if not heatmap_maxt is None else N_spectra*time_delay
		extent = tx[0], tx[1], 0, maxt

		tempy = spectra[np.logical_and(tx[0] < spectrax, spectrax < tx[1])]
		if heatmap_maxt is not None:
			tempy = tempy[:,:heatmap_maxt]

		plt.ylabel('Time (s)')
		plt.xlabel('$\lambda$ (nm)')
		plt.imshow(tempy.T, extent=extent, aspect='auto', origin='lower')
		cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(spectra.min(), spectra.max()), cmap=cmap))
		cbar.set_label('Absorbance (a.u.)')
		plt.savefig(f'{plots_dir}/spectra_heatmap_kinetics_{i}.jpg')

	plt.close('all')

	
	def get_tracking_absorbance(wavelenght):
		low = wavelenght[0]
		high = wavelenght[1]

		prev_wl = 0
		for i, wl in enumerate(spectrax):
			if prev_wl <= low and wl >= low:
				low_i = i-1
			elif prev_wl <= high and wl >= high:
				high_i = i-1

			prev_wl = wl

		return np.mean(spectra[low_i:high_i,:], axis=0)

	tracking_spectra = []
	for twl in tracking_spectrax:
		absorbs = get_tracking_absorbance(twl)
		# print(np.where(absorbs < 0.001))
		absorbs = np.delete(absorbs, np.where(absorbs < 0.001))
		tracking_spectra.append(absorbs)

		if type(twl) is tuple: twlb = (twl[0]+twl[1])/2
		else: twlb = twl

		plt.scatter(np.ones_like(absorbs) * twlb, absorbs)

	plt.figure()
	plt.title('Tracked spectrax')
	plt.xlabel('t (s)')
	plt.ylabel('Absorbance (a.u.)')
	for twl, tabs in zip(tracking_spectrax, tracking_spectra):
		plt.plot(np.arange(len(tabs))*time_delay, tabs, label=r'$\lambda_{avg}$' + f' over [{twl[0]}, {twl[1]}] nm')
	plt.legend()
	plt.savefig(f'{plots_dir}/tracked_spectra.jpg')

	plt.figure()
	plt.title('Tracked spectrax (log-scale)')
	plt.xlabel('t (s)')
	plt.ylabel('ln(Absorbance)')
	for twl, tabs in zip(tracking_spectrax, tracking_spectra):
		plt.plot(np.arange(len(tabs))*time_delay, np.log(tabs), label=r'$\lambda_{avg}$' + f' over [{twl[0]}, {twl[1]}] nm')
	plt.legend()
	plt.savefig(f'{plots_dir}/tracked_spectra_log.jpg')

	# for twl, tabs in zip(tracking_spectrax, tracking_spectra):
	# 	low = twl[0]
	# 	high = twl[1]

		# fit_exp_decay(np.arange(len(tabs))*time_delay, np.asarray(tabs), title=f'Predicted and reference rate (around $\lambda ∈ [{low}, {high}]$ nm)', plots_dir=plots_dir, use_scipy=True)


	# plt.show()	
	np.save(f'./{name}_array.npy', tracking_spectra)
	logfile.close()
	# print('done')
	return tracking_spectra


if __name__ == '__main__':
	settings = {
		'SP_acetone_UVVIS_20s_180x_5um': {
			'name': 'kinetics_acetone_20s_180x_5um_1',
			'file': 'data/20211116_UVVIS_sp_aceton_5um_20_180/data.csv',
			'time_delay': 20,
			'tracking_spectrax': [(520, 580)],
		},
		'SP_cyclohexanone_UVVIS_10s_360x_20um': {
			'name': 'kinetics_cyclohexanone_20s_180x_5um_1',
			'file': "data/20211116_UVVIS_sp_cyclohexanon_20um/data_kinetics_cyclohexanone_10s_360x_20um.csv",
			'time_delay': 10,
			'tracking_spectrax': [(520, 580)],
		},
		'SP_cyclohexanone_UVVIS_10s_360x_20um_decoupled': {
			'name': 'kinetics_cyclohexanone_20s_180x_5um_decoupled',
			'file': "data/20211116_UVVIS_sp_cyclohexanon_20um/data_kinetics_cyclohexanone_10s_360x_20um_decoupled.csv",
			'time_delay': 10,
			'tracking_spectrax': [(520, 580)],
		},
		'SP_cyclohexanone_UVVIS_10s_360x_20um_decoupled_2': {
			'name': 'kinetics_cyclohexanone_20s_180x_5um_decoupled_2',
			'file': "data/20211118_UVVIS_sp_cyclohexanon_uncoupled/data.csv",
			'time_delay': 10,
			'tracking_spectrax': [(520, 580)],
			'spectra_blacklist': [61, 285]
		},
		'SP_cyclohexanone_UVVIS_10s_360x_20um_coupled_2': {
			'name': 'kinetics_cyclohexanone_20s_180x_5um_coupled_2',
			'file': "data/20211119_UVVIS_sp_cyclohexanon_coupled/data.csv",
			'time_delay': 10,
			'tracking_spectrax': [(520, 580)],
		},
		'SP_cyclohexane_10s_360x_5um_coupled': {
			'name': 'kinetics_cycolhexanone_10s_360x_5um_coupled_1',
			'file': "data/20211126_UVVIS_SP_cyclohexanone_5um_coupled/data.csv",
			'time_delay': 10,
			'tracking_spectrax': [(520, 580)],
		},
		'SP_cyclohexane_10s_360x_5um_decoupled': {
			'name': 'kinetics_cycolhexanone_10s_360x_5um_decoupled_1',
			'file': "data/20211126_UVVIS_SP_cyclohexanone_5um_decoupled/data.csv",
			'time_delay': 10,
			'tracking_spectrax': [(520, 580)],
		},
		'20211129_coupled_1': {
			'name': '20211129_coupled_1',
			'file':"data/20211129_coupled_kinetics_CHXO_1/kinetics.csv",
			'time_delay': 10,
			'tracking_spectrax': [(520, 580)],
		},
		'20211129_decoupled_1': {
			'name': '20211129_decoupled_1',
			'file':"data/20211129_decoupled_kinetics_CHXO_1/kinetics.csv",
			'time_delay': 10,
			'tracking_spectrax': [(520, 580)],
		}
	}

	use_coupled_biexp_decay = True
	use_uncoupled_biexp_decay = True


	# cy = [main(**settings['SP_cyclohexanone_UVVIS_10s_360x_20um'])[0],
	# 	  main(**settings['SP_cyclohexanone_UVVIS_10s_360x_20um_coupled_2'])[0],]
	# uncy = [main(**settings['SP_cyclohexanone_UVVIS_10s_360x_20um_decoupled'])[0],
	# 		main(**settings['SP_cyclohexanone_UVVIS_10s_360x_20um_decoupled_2'])[0]]

	# cy   = [main(**settings['SP_cyclohexane_10s_360x_5um_coupled'])[0],]
	# uncy = [main(**settings['SP_cyclohexane_10s_360x_5um_decoupled'])[0],]

	cy   = [main(**settings['20211129_coupled_1'])[0],]
	uncy = [main(**settings['20211129_decoupled_1'])[0],]

	cx = [np.arange(yy.size)*10 for yy in cy]
	uncx = [np.arange(yy.size)*10 for yy in uncy]

	plt.close('all')
	plt.figure()
	plt.title(r'Kinetics of average signal between 520 nm and 580 nm')
	plt.xlabel('t (s)')
	plt.ylabel('Absorbance (a.u.)')
	for i, x, y in zip(range(len(cx)), cx, cy):
		plt.plot(x, y, label=f'Coupled cyclohexanone {i}')
	for i, x, y in zip(range(len(uncx)), uncx, uncy):
		plt.plot(x, y, label=f'Uncoupled cyclohexanone {i}')
	plt.legend()

	plt.figure()
	plt.title(r'Normalised kinetics of average signal between 520 nm and 580 nm')
	plt.xlabel('t (s)')
	plt.ylabel('Absorbance (normalised)')
	for i, x, y in zip(range(len(cx)), cx, cy):
		plt.plot(x, (y-y.min())/(y.max()-y.min()), label=f'Coupled, experiment {i}')
	for i, x, y in zip(range(len(uncx)), uncx, uncy):
		plt.plot(x, (y-y.min())/(y.max()-y.min()), label=f'Uncoupled, experiment {i}')
		
	if use_coupled_biexp_decay:
		coupled_fit   = [fit_biexp_decay(x, y - np.mean(y[-20:-1]), show_fit=True) for x, y in zip(cx, cy)]
	else:
		coupled_fit   = [fit_exp_decay(x, y - np.mean(y[-20:-1]),   show_fit=True) for x, y in zip(cx, cy)]

	if use_uncoupled_biexp_decay:
		uncoupled_fit = [fit_biexp_decay(x, y - np.mean(y[-20:-1]), show_fit=True) for x, y in zip(uncx, uncy)]
	else:
		uncoupled_fit = [fit_exp_decay(x, y - np.mean(y[-20:-1]),   show_fit=True) for x, y in zip(uncx, uncy)]

	coupled_halftimes = []
	print('\n======Coupled experiments=========')
	for i, c in zip(range(len(coupled_fit)), coupled_fit):
		if use_coupled_biexp_decay:
			print(f'Coupled biexponential fitting for experiment {i}:')
			print(f'\tCorrelation (R2)  = {c["r2_value"]:.10f}')
			print(f'\tk1                = {c["k1"]:.8f} s^-1')
			print(f'\tk2                = {c["k2"]:.8f} s^-1')
			print(f'\tA01               = {c["A01"]:.3f}')
			print(f'\tA02               = {c["A02"]:.3f}')
			print(f'\tB                 = {c["B"]:.3f}')
			print(f'\tLife-time 1       = {1/c["k1"]:.2f} s')
			print(f'\tLife-time 2       = {1/c["k2"]:.2f} s')
			print(f'\tHalf-life 1       = {np.log(2)/c["k1"]:.2f} s')
			print(f'\tHalf-life 2       = {np.log(2)/c["k2"]:.2f} s')

			coupled_halftimes.append((np.log(2)/c["k1"], np.log(2)/c["k2"]))

		else:
			print(f'Coupled exponential fitting for experiment {i}:')
			print(f'\tCorrelation (R2)  = {c["r2_value"]:.10f}')
			print(f'\tk                 = {c["k"]:.8f} s^-1')
			print(f'\tA0                = {c["A0"]:.3f}')
			# print(f'\tB                 = {c["B"]:.3f}')
			print(f'\tLife-time         = {1/c["k"]:.2f} s')
			print(f'\tHalf-life         = {np.log(2)/c["k"]:.2f} s')
			
			coupled_halftimes.append(np.log(2)/c["k"])


	print('\n=====Uncoupled experiments========')
	uncoupled_halftimes = []
	for i, c in zip(range(len(uncoupled_fit)), uncoupled_fit):
		if use_uncoupled_biexp_decay:	
			print(f'Uncoupled biexponential fitting for experiment {i}:')
			print(f'\tCorrelation (R2)  = {c["r2_value"]:.10f}')
			print(f'\tk1                = {c["k1"]:.8f} s^-1')
			print(f'\tk2                = {c["k2"]:.8f} s^-1')
			print(f'\tA01               = {c["A01"]:.3f}')
			print(f'\tA02               = {c["A02"]:.3f}')
			print(f'\tB                 = {c["B"]:.3f}')
			print(f'\tLife-time 1       = {1/c["k1"]:.2f} s')
			print(f'\tLife-time 2       = {1/c["k2"]:.2f} s')
			print(f'\tHalf-life 1       = {np.log(2)/c["k1"]:.2f} s')
			print(f'\tHalf-life 2       = {np.log(2)/c["k2"]:.2f} s')

			uncoupled_halftimes.append((np.log(2)/c["k1"], np.log(2)/c["k2"]))
		else:
			print(f'Uncoupled exponential fitting for experiment {i}:')
			print(f'\tCorrelation (R2)  = {c["r2_value"]:.10f}')
			print(f'\tk                 = {c["k"]:.8f} s^-1')
			print(f'\tA0                = {c["A0"]:.3f}')
			# print(f'\tB                 = {c["B"]:.3f}')
			print(f'\tLife-time         = {1/c["k"]:.2f} s')
			print(f'\tHalf-life         = {np.log(2)/c["k"]:.2f} s')
			
			uncoupled_halftimes.append(np.log(2)/c["k"])

	print('\n============Summary===============')
	print(f'Number of experiments:')
	print(f'\tCoupled:   {len(cx)}')
	print(f'\tUncoupled: {len(uncx)}')

	print(f'\nCoupled half-times:')
	if use_coupled_biexp_decay:
		for i, h in enumerate(coupled_halftimes):
			print(f'\tExperiment {i}: {h[0]:.2f} s and {h[1]:.2f} s')

		average_h1 = sum([h[0] for h in coupled_halftimes])/len(coupled_halftimes)
		average_h2 = sum([h[1] for h in coupled_halftimes])/len(coupled_halftimes)
		stddev_h1 = np.std(coupled_halftimes, axis=0)[0]
		stddev_h2 = np.std(coupled_halftimes, axis=0)[1]
		print(f'\tAverage     : {average_h1:.2f} s and {average_h2:.2f} s')
		print(f'\tStddev      : {stddev_h1:.2f} s and {stddev_h2:.2f} s')
	else:
		for i, h in enumerate(coupled_halftimes):
			print(f'\tExperiment {i}: {h:.2f} s')
		average_h = sum(coupled_halftimes)/len(coupled_halftimes)
		stddev_h = np.std(coupled_halftimes)
		print(f'\tAverage     : {average_h:.2f} s')
		print(f'\tStddev      : {stddev_h:.2f} s')

	print(f'\nUncoupled half-times:')
	if use_coupled_biexp_decay:
		for i, h in enumerate(uncoupled_halftimes):
			print(f'\tExperiment {i}: {h[0]:.2f} s and {h[1]:.2f} s')
		average_h1 = sum([h[0] for h in uncoupled_halftimes])/len(uncoupled_halftimes)
		average_h2 = sum([h[1] for h in uncoupled_halftimes])/len(uncoupled_halftimes)
		stddev_h1 = np.std(uncoupled_halftimes, axis=0)[0]
		stddev_h2 = np.std(uncoupled_halftimes, axis=0)[1]
		print(f'\tAverage     : {average_h1:.2f} s and {average_h2:.2f} s')
		print(f'\tStddev      : {stddev_h1:.2f} s and {stddev_h2:.2f} s')
	else:
		for i, h in enumerate(uncoupled_halftimes):
			print(f'\tExperiment {i}: {h:.2f} s')
		average_h = sum(uncoupled_halftimes)/len(uncoupled_halftimes)
		stddev_h = np.std(uncoupled_halftimes)
		print(f'\tAverage     : {average_h:.2f} s')
		print(f'\tStddev      : {stddev_h:.2f} s')

	plt.show()