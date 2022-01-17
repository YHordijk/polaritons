
'''
Script used to analyse scanning measurements across wavelengths
Given a path to a directory with data (data_dir)
It expects to find:
data_dir/0/, data_dir/1/, ...
data_dir/meta.txt
data_dir/i/cavity.csv
data_dir/i/kinetics.csv
data_dir/i/meta.txt

It will analyse the results and output to results/main/name (res_dir) which contains
res_dir/i/python.log
res_dir/i/plots/*
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
	main_data_dir = "data/SP_scan_new_cavity2"
	mirror_less_data_dir = "data/no_au"
	n = 1.3008 #refracative index of cyclohexanone
	target_wn = 1200

	
	#### setup
	data_dirs = {p: f'{main_data_dir}/{p}' for p in os.listdir(main_data_dir) if (os.path.isdir(f'{main_data_dir}/{p}') and not p.startswith('!'))}
	mirror_less_data_dirs = {p: f'{mirror_less_data_dir}/{p}' for p in os.listdir(mirror_less_data_dir) if (os.path.isdir(f'{mirror_less_data_dir}/{p}') and not p.startswith('!'))}

	name = main_data_dir.split('/')[1]
	main_res_dir = f"results/{name}"
	if not os.path.exists(main_res_dir):
		os.mkdir(main_res_dir)
	main_plots_dir = f'{main_res_dir}/plots'
	if not os.path.exists(main_plots_dir):
		os.mkdir(main_plots_dir)

	main_python_log = f'{main_res_dir}/python.log'
	main_logfile = open(main_python_log, 'w')

	main_results = {}
	print(f'====== BEGINNING SCANNING ANALYSIS ======', file=main_logfile)
	print(f'Main data directory: {main_data_dir}', file=main_logfile)
	Nexp = len(data_dirs.keys())
	Nno_au = len(mirror_less_data_dirs.keys())
	print(f'Found {Nexp} experiments!', file=main_logfile)
	i = 0
	for experiment, data_dir in data_dirs.items():
		print(f'\t{i: <3} {experiment:<6}: {data_dir}', file=main_logfile)
		i += 1

	print(f'Found {Nno_au} experiments!', file=main_logfile)
	i = 0
	for experiment, data_dir in mirror_less_data_dirs.items():
		print(f'\t{i: <3} {experiment:<6}: {data_dir}', file=main_logfile)
		i += 1

	cmap = plt.get_cmap('tab10')
	exp_colors = {e: cmap(i/Nexp) for i,e in enumerate(data_dirs.keys())}

	for experiment, data_dir in data_dirs.items():
		main_results[experiment] = {}
		#set paths
		res_dir = f'results/{name}/{experiment}'

		if not os.path.exists(res_dir):
			os.mkdir(res_dir)

		cavity_path = f'{data_dir}/cavity.csv'
		kinetic_path = f'{data_dir}/kinetics.csv'
		meta_path = f'{data_dir}/meta.txt'
		python_log = f'{res_dir}/python.log'

		#begin experiment analysis
		with open(python_log, 'w') as f:
			f.write('====== STARTING ANALYSIS ======\n')
			f.write(f'Data directory:    {data_dir}\n')
			f.write(f'Results directory: {res_dir}\n')

			if os.path.exists(meta_path):
				f.write(f'\nMeta comments:\n')
				with open(meta_path, 'r') as meta:
					f.write(''.join(meta.readlines()))


			f.write('\n====== BEGINNING CAVITY ANALYSIS\n')

		if os.path.exists(cavity_path):	
			cavity_settings = {
				'file': cavity_path,
				'name': name,
				'logfile': python_log,
				'polariton_wns': [],
				'refractive_index': n,
				'plot_fringes_for_one': True,
				'plot_polaritons': True,
				'plot_peaks': True,
				'overlay_spectrum_path': "data/cyclohexanone_3um.csv",
				'result_dir': res_dir,
			}


			aIR_results = aIR.main(**cavity_settings, show_plots=False)

			main_results[experiment]['FSR'] = aIR_results['FSRs'][0]
			main_results[experiment]['FSR_offset'] = aIR_results['FSR_offsets'][0]
			main_results[experiment]['FSR_error'] = aIR_results['FSR_stds'][0]
		else:
			with open(python_log, 'a') as f: f.write(f'{cavity_path} was not found/n')

		with open(python_log, 'a') as f:
			f.write('\n====== BEGINNING KINETICS ANALYSIS\n')

		logfile = open(python_log, 'a')
		if os.path.exists(kinetic_path):
			kinetics_settings = {
				'file': kinetic_path,
				'name': name,
				'logfile': python_log,
				'result_dir': res_dir,
				'time_delay': 10,
				'tracking_spectrax': [(520, 580)],
				'heatmap_maxt': 60,
			}

			
			cy = aKIN.main(**kinetics_settings, show_plots=False)
			cx = [np.arange(yy.size)*10 for yy in cy]

			main_results[experiment]['ys'] = [y - np.mean(y[-20:-1]) for y in cy]
			main_results[experiment]['x'] = cx

			fits = []
			for i, y, x in zip(range(len(cy)), cy, cx):
				if experiment in ['2']: #specify here if kinetics should be fitted using biexp or exp
					fits.append(exp_decay_fitter.fit_exp_lin(x, y - np.mean(y[-20:-1]), plots_dir=res_dir+'/plots', index=i))
				else:	
					fits.append(exp_decay_fitter.fit_exp_lin(x, y - np.mean(y[-20:-1]), plots_dir=res_dir+'/plots', index=i))

			main_results[experiment]['fits'] = fits

			print('=== KINETICS', file=logfile)
			for w, c, x in zip(kinetics_settings['tracking_spectrax'], fits, cx):
				if c['name'] == 'biexp_decay':
					print(f'Biexponential fitting for average between wavelenghts [{w[0]:.1f}, {w[1]:.1f}] nm:', file=logfile)
					print(f'\tt in [{x.min()}, {x.max()}] s', file=logfile)
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

					main_results[experiment]['k1'] = c['k1']
					main_results[experiment]['k2'] = c['k2']

				if c['name'] == 'exp_decay':
					print(f'Exponential fitting for average between wavelenghts [{w[0]:.1f}, {w[1]:.1f}] nm:', file=logfile)
					print(f'\tt in [{x.min()}, {x.max()}] s', file=logfile)
					print(f'\tCorrelation (R2)  = {c["r2_value"]:.10f}', file=logfile)
					print(f'\tk                 = {c["k"]:.8f} s^-1', file=logfile)
					print(f'\tA0                = {c["A0"]:.3f}', file=logfile)
					print(f'\tB                 = {c["B"]:.3f}', file=logfile)
					print(f'\tLife-time         = {1/c["k"]:.2f} s', file=logfile)
					print(f'\tHalf-life         = {np.log(2)/c["k"]:.2f} s', file=logfile)

					main_results[experiment]['k1'] = c['k']

				if c['name'] == 'exp_lin':
					print(f'Exponential fitting for average between wavelenghts [{w[0]:.1f}, {w[1]:.1f}] nm:', file=logfile)
					print(f'\tt in [{x.min()}, {x.max()}] s', file=logfile)
					print(f'\tCorrelation (R2)  = {c["r2_value"]:.10f}', file=logfile)
					print(f'\tk                 = {c["k"]:.8f} s^-1', file=logfile)
					print(f'\tA0                = {c["A0"]:.3f}', file=logfile)
					print(f'\tslope             = {c["B"]:.3f}', file=logfile)
					print(f'\tB                 = {c["B"]:.3f}', file=logfile)
					print(f'\tLife-time         = {1/c["k"]:.2f} s', file=logfile)
					print(f'\tHalf-life         = {np.log(2)/c["k"]:.2f} s', file=logfile)

					main_results[experiment]['k1'] = c['k']

			print('=== END', file=logfile)
		else:
			print(f'{kinetic_path} was not found', file=logfile)
		print('\n====== END', file=logfile)
		logfile.close()

	if all('FSR' in r for r in main_results.values()):
		print('FSR succesfully calculated for all experiments!', file=main_logfile)
	else:
		print('FSR calculation failed for following experiments:', file=main_logfile)
		for e, r in main_results.items():
			if 'FSR' not in r:
				print(f'\t{e}', file=main_logfile)

	if all('k1' in r for r in main_results.values()):
		print('k1 succesfully calculated for all experiments!', file=main_logfile)
	else:
		print('k1 calculation failed for following experiments:', file=main_logfile)
		for e, r in main_results.items():
			if 'k1' not in r:
				print(f'\t{e}', file=main_logfile)

	mirrorless_results = {}
	if Nno_au > 0:
		for experiment, data_dir in mirror_less_data_dirs.items():
			mirrorless_results[experiment] = {}
			#set paths
			res_dir = f'results/{name}/{experiment}'

			if not os.path.exists(res_dir):
				os.mkdir(res_dir)

			kinetic_path = f'{data_dir}/kinetics.csv'
			meta_path = f'{data_dir}/meta.txt'
			python_log = f'{res_dir}/python.log'

			#begin experiment analysis
			with open(python_log, 'w') as f:
				f.write('====== STARTING ANALYSIS ======\n')
				f.write(f'Data directory:    {data_dir}\n')
				f.write(f'Results directory: {res_dir}\n')

				if os.path.exists(meta_path):
					f.write(f'\nMeta comments:\n')
					with open(meta_path, 'r') as meta:
						f.write(''.join(meta.readlines()))


			with open(python_log, 'a') as f:
				f.write('\n====== BEGINNING KINETICS ANALYSIS\n')

			logfile = open(python_log, 'a')
			if os.path.exists(kinetic_path):
				kinetics_settings = {
					'file': kinetic_path,
					'name': name,
					'logfile': python_log,
					'result_dir': res_dir,
					'time_delay': 10,
					'tracking_spectrax': [(520, 580)],
					'heatmap_maxt': 60,
				}

				
				cy = aKIN.main(**kinetics_settings, show_plots=False)
				cx = [np.arange(yy.size)*10 for yy in cy]

				mirrorless_results[experiment]['ys'] = [y - np.mean(y[-20:-1]) for y in cy]
				mirrorless_results[experiment]['x'] = cx

				fits = []
				for i, y, x in zip(range(len(cy)), cy, cx):
					if experiment in ['2']: #specify here if kinetics should be fitted using biexp or exp
						fits.append(exp_decay_fitter.fit_exp_lin(x, y - np.mean(y[-20:-1]), plots_dir=res_dir+'/plots', index=i))
					else:	
						fits.append(exp_decay_fitter.fit_exp_lin(x, y - np.mean(y[-20:-1]), plots_dir=res_dir+'/plots', index=i))

				mirrorless_results[experiment]['fits'] = fits

				print('=== KINETICS', file=logfile)
				for w, c, x in zip(kinetics_settings['tracking_spectrax'], fits, cx):
					if c['name'] == 'biexp_decay':
						print(f'Biexponential fitting for average between wavelenghts [{w[0]:.1f}, {w[1]:.1f}] nm:', file=logfile)
						print(f'\tt in [{x.min()}, {x.max()}] s', file=logfile)
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

						mirrorless_results[experiment]['k1'] = c['k1']
						mirrorless_results[experiment]['k2'] = c['k2']

					if c['name'] == 'exp_decay':
						print(f'Exponential fitting for average between wavelenghts [{w[0]:.1f}, {w[1]:.1f}] nm:', file=logfile)
						print(f'\tt in [{x.min()}, {x.max()}] s', file=logfile)
						print(f'\tCorrelation (R2)  = {c["r2_value"]:.10f}', file=logfile)
						print(f'\tk                 = {c["k"]:.8f} s^-1', file=logfile)
						print(f'\tA0                = {c["A0"]:.3f}', file=logfile)
						print(f'\tB                 = {c["B"]:.3f}', file=logfile)
						print(f'\tLife-time         = {1/c["k"]:.2f} s', file=logfile)
						print(f'\tHalf-life         = {np.log(2)/c["k"]:.2f} s', file=logfile)

						mirrorless_results[experiment]['k1'] = c['k']

					if c['name'] == 'exp_lin':
						print(f'Exponential fitting for average between wavelenghts [{w[0]:.1f}, {w[1]:.1f}] nm:', file=logfile)
						print(f'\tt in [{x.min()}, {x.max()}] s', file=logfile)
						print(f'\tCorrelation (R2)  = {c["r2_value"]:.10f}', file=logfile)
						print(f'\tk                 = {c["k"]:.8f} s^-1', file=logfile)
						print(f'\tA0                = {c["A0"]:.3f}', file=logfile)
						print(f'\tslope             = {c["B"]:.3f}', file=logfile)
						print(f'\tB                 = {c["B"]:.3f}', file=logfile)
						print(f'\tLife-time         = {1/c["k"]:.2f} s', file=logfile)
						print(f'\tHalf-life         = {np.log(2)/c["k"]:.2f} s', file=logfile)

						mirrorless_results[experiment]['k1'] = c['k']

				print('=== END', file=logfile)
			else:
				print(f'{kinetic_path} was not found', file=logfile)
			print('\n====== END', file=logfile)
			logfile.close()

		if all('k1' in r for r in mirrorless_results.values()):
			print('k1 succesfully calculated for all experiments!', file=main_logfile)
		else:
			print('k1 calculation failed for following experiments:', file=main_logfile)
			for e, r in mirrorless_results.items():
				if 'k1' not in r:
					print(f'\t{e}', file=main_logfile)


	for e, r in main_results.items():
		r['spacing'] = 10_000/(2*n*r['FSR'])
		r['tuned v'] = sorted((r['FSR']*i for i in range(20)), key=lambda f: abs(f-target_wn))[0]
	spacings = {e: 10_000/(2*n*r['FSR']) for e, r in main_results.items() if 'FSR' in r}
	tuned_wn = {e: sorted((r['FSR']*i for i in range(20)), key=lambda f: abs(f-target_wn))[0] for e, r in main_results.items()}


	print(f'\nResults:', file=main_logfile)
	print(f'\tExperiment | FSR (cm^-1) | v (cm^-1) | Spacing (um) | t (s)', file=main_logfile)
	i = 0
	for e, r in main_results.items():
		print(f'\t{i: <10} | {r["FSR"]: >11.2f} | {r["tuned v"]: >9.2f} | {r["spacing"]: 12.2f} | {1/r["k1"]: 8.3f}', file=main_logfile)
		i+=1
	print(f'\nResults (mirrorless):', file=main_logfile)
	print(f'\tExperiment | t (s)', file=main_logfile)
	i = 0
	for e, r in mirrorless_results.items():
		print(f'\t{i: <10} | {1/r["k1"]: 8.3f}', file=main_logfile)
		i+=1



	cyclohexanone, _ = aIR.read_csv("data/cyclohexanone_3um.csv")
	spiropyran, _ = aIR.read_csv("data/SP_computed.csv")
	merocyanine, _ = aIR.read_csv("data/MC_computed.csv")

	cx = cyclohexanone[:,0]
	cy = 10**-cyclohexanone[:,1]
	cy = cyclohexanone[:,1]

	spx = spiropyran[:,0]
	spy = spiropyran[:,1]

	mcx = merocyanine[:,0]
	mcy = merocyanine[:,1]

	cy = cy/cy.max()
	spy = spy/spy.max()
	mcy = mcy/mcy.max()


	for k in ['k1', 'k2']:
		try:
			plt.close('all')
			f, ax = plt.subplots()
			ax2 = plt.twinx(ax)
			ax.plot(cx, cy, label='IR spectrum cyclohexanone')
			ax.plot(mcx, mcy, label='IR spectrum nitro-merocyanine')
			ax.plot(spx, spy, label='IR spectrum nitro-spiropyran')
			ax2.scatter(tuned_wn.values(), [r[k]*1000 if k in r else 0 for e, r in main_results.items()], c=list(exp_colors.values()))
			ax2.errorbar(tuned_wn.values(), [r[k]*1000 if k in r else 0 for e, r in main_results.items()], xerr=[r['FSR_error'] if k in r else 0 for e, r in main_results.items()], fmt='none')
			ax.set_xlim(min(tuned_wn.values())-30, max(tuned_wn.values())+30)
			ax2.spines['right'].set_color('red')
			ax2.yaxis.label.set_color('red')
			ax2.tick_params(axis='y', colors='red')
			if Nno_au > 0:
				no_au_mean = np.asarray([r[k]*1000 if k in r else 0 for e, r, in mirrorless_results.items()]).mean()
				no_au_sd = np.std(np.asarray([r[k]*1000 if k in r else 0 for e, r, in mirrorless_results.items()]))
				# print(no_au_sd)
				ax2.plot((min(tuned_wn.values())-30, max(tuned_wn.values())+30), (no_au_mean, no_au_mean), color='black', linestyle='dashed', label='Mirrorless rate (5um) + SD')
				ax2.fill_between((min(tuned_wn.values())-30, max(tuned_wn.values())+30), no_au_mean-no_au_sd, no_au_mean+no_au_sd, color='black', alpha=0.25)
			ax2.set_xlim(min(tuned_wn.values())-30, max(tuned_wn.values())+30)
			ax2.set_ylim(0, max(r[k]*1.2*1000 for r in main_results.values()))
			ax.set_xlabel(r'$\nu (cm^{-1})$')
			ax.set_ylabel('Transmission')
			ax2.set_ylabel(rf'Reaction rate ({k}) $(10^{-3}s^{-1})$')
			f.legend()
			plt.tight_layout()
			plt.savefig(f'{main_plots_dir}/{k}_scanning.png')
		except:
			continue

	plt.figure()
	plt.ylabel('Absorbance (a.u.)')
	plt.xlabel('t (s)')
	i = 0
	for e, r in main_results.items():
		plt.scatter(r['x'][0], r['ys'][0], color=exp_colors[e], alpha=.5, s=2)
		plt.plot(r['x'][0], r['fits'][0]['ypred'], color=exp_colors[e], label=f'Experiment {i}, $r^2$={r["fits"][0]["r2_value"]}')
		i += 1

	plt.tight_layout()
	plt.legend()
	plt.savefig(f'{main_plots_dir}/all_kinetics.png')

	plt.figure()
	plt.ylabel('Absorbance (a.u.)')
	plt.xlabel('t (s)')
	i = 0
	for e, r in mirrorless_results.items():
		plt.scatter(r['x'][0], r['ys'][0], alpha=.5, s=2)
		plt.plot(r['x'][0], r['fits'][0]['ypred'], label=f'Experiment {i}, $r^2$={r["fits"][0]["r2_value"]}')
		i += 1

	plt.tight_layout()
	plt.legend()
	plt.savefig(f'{main_plots_dir}/all_kinetics_mirrorless.png')

	plt.figure()
	plt.ylabel('Absorbance (normalized)')
	plt.xlabel('t (s)')
	i = 0
	for e, r in main_results.items():
		x = r['x'][0]
		y = r['ys'][0]
		lin = r['fits'][0]['b'] * x + r['fits'][0]['B']
		ypred = r['fits'][0]['ypred']

		#normalize them
		y = (y-lin)/(y-lin).max()
		ypred = (ypred-lin)/(ypred-lin).max()
		plt.scatter(x, y, color=exp_colors[e], alpha=.5, s=2)
		plt.plot(x, ypred, color=exp_colors[e], label=f'Experiment {i}, $t$={1/r["fits"][0]["k"]:.2f}s')
		i += 1

	plt.tight_layout()
	plt.legend()
	plt.savefig(f'{main_plots_dir}/all_kinetics_corrected_normed.png')

	plt.figure()
	plt.ylabel('Absorbance (normalized)')
	plt.xlabel('t (s)')
	i = 0
	for e, r in mirrorless_results.items():
		x = r['x'][0]
		y = r['ys'][0]
		lin = r['fits'][0]['b'] * x + r['fits'][0]['B']
		ypred = r['fits'][0]['ypred']

		#normalize them
		y = (y-lin)/(y-lin).max()
		ypred = (ypred-lin)/(ypred-lin).max()
		plt.scatter(x, y, alpha=.5, s=2)
		plt.plot(x, ypred, label=f'Experiment {i}, $t$={1/r["fits"][0]["k"]:.2f}s')
		i += 1

	plt.tight_layout()
	plt.legend()
	plt.savefig(f'{main_plots_dir}/all_kinetics_corrected_normed.png')

	# plt.show()