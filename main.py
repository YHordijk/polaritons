
'''
Script used to analyse completed kinetics measurements
Given a path to a directory with data (data_dir)
It expects to find:
data_dir/cavity.csv
data_dir/kinetics.csv
data_dir/meta.txt

It will analyse the results and output to results/main/name (res_dir) which contains
res_dir/plots/*
res_dir/python.log
'''


import os
import numpy as np
import matplotlib.pyplot as plt
import analysis_IR_2 as aIR
import analysis_kinetics as aKIN
import pkg.exp_decay_fitter as exp_decay_fitter


if __name__ == '__main__':
	data_dir = "data/20211129_coupled_kinetics_CHXO_1"

	#### setup
	name = data_dir.split('/')[1]
	res_dir = f"results/{name}"

	if not os.path.exists(res_dir):
		os.mkdir(res_dir)

	cavity_path = f'{data_dir}/cavity.csv'
	kinetic_path = f'{data_dir}/kinetics.csv'
	meta_path = f'{data_dir}/meta.txt'
	python_log = f'{res_dir}/python.log'



	with open(python_log, 'w') as f:
		f.write('====== STARTING ANALYSIS\n')
		f.write(f'Data directory:    {data_dir}\n')
		f.write(f'Results directory: {res_dir}\n')
		f.write(f'\nMeta comments:\n')
		with open(meta_path, 'r') as meta:
			f.write(''.join(meta.readlines()) + '\n')

		f.write('====== BEGINNING CAVITY ANALYSIS\n')

	cavity_settings = {
		'file': cavity_path,
		'name': name,
		'logfile': python_log,

		'polariton_wns': [1713],
		'refractive_index': 1.4507,
		'plot_fringes_for_one': True,
		'plot_polaritons': True,
		'plot_peaks': True,
		'overlay_spectrum_path': "data/cyclohexanone_3um.csv",
		'result_dir': res_dir,
	}

	aIR.main(**cavity_settings, show_plots=False)



	with open(python_log, 'a') as f:
		f.write('\n====== BEGINNING KINETICS ANALYSIS\n')

	kinetics_settings = {
		'file': kinetic_path,
		'name': name,
		'logfile': python_log,
		'result_dir': res_dir,
		'time_delay': 10,
		'tracking_spectrax': [(520, 580)],
		'spectra_blacklist': [0], 

		'heatmap_maxt': 60,
	}

	
	cy = aKIN.main(**kinetics_settings, show_plots=False)
	logfile = open(python_log, 'a')
	cx = np.array([np.arange(yy.size)*10 for yy in cy]).flatten()

	fits = []
	for i, y in enumerate(cy):
		fits.append(exp_decay_fitter.fit_biexp_decay(cx, y - np.mean(y[-20:-1]), plots_dir=res_dir+'/plots', index=i))

	print('=== KINETICS', file=logfile)
	print(f'Model: A01*exp(-k1*t) + A02*exp(-k2*t) + B,   t in [{cx.min()}, {cx.max()}] s', file=logfile)
	for w, c in zip(kinetics_settings['tracking_spectrax'], fits):
		print(f'Biexponential fitting for wavelength around {w[0]/2 + w[1]/2:.1f} nm:', file=logfile)
		print(f'\tCorrelation (R2)  = {c["r2_value"]:.10f}', file=logfile)
		print(f'\tk1                = {c["k1"]:.8f} s^-1', file=logfile)
		print(f'\tk2                = {c["k2"]:.8f} s^-1', file=logfile)
		print(f'\tA01               = {c["A01"]:.3f}', file=logfile)
		print(f'\tA02               = {c["A02"]:.3f}', file=logfile)
		print(f'\tB                 = {c["B"]:.3f}', file=logfile)
		print(f'\tLife-time 1       = {1/c["k1"]:.2f} s', file=logfile)
		print(f'\tLife-time 2       = {1/c["k2"]:.2f} s', file=logfile)
		print(f'\tHalf-life 1       = {np.log(2)/c["k1"]:.2f} s', file=logfile)
		print(f'\tHalf-life 2       = {np.log(2)/c["k2"]:.2f} s', file=logfile)

	print('=== END', file=logfile)
	print('\n====== END', file=logfile)
	logfile.close()