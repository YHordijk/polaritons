import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from pkg.exp_decay_fitter import fit_exp_decay
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


	if not os.path.exists(f'results/{name}'):
		os.mkdir(f'results/{name}')
	logfile = open(f'results/{name}/python.log', 'w')
	plots_dir = f'results/{name}/plots'
	if not os.path.exists(plots_dir):
		os.mkdir(plots_dir)

	spectrax = data[:,0]
	spectra = data[:,1:]
	min_abs = min([a.min() for a in spectra])
	max_abs = max([a.max() for a in spectra])
	wavelenght_stepsize = abs(np.diff(spectrax).mean())
	
	N_spectra = spectra.shape[1]
	
	print(f'Loaded {N_spectra} spectra!', file=logfile)
	print(f'labda_range  =  [{spectrax[0]}, {spectrax[-1]}] nm', file=logfile)
	print(f'deltav   =  {wavelenght_stepsize:.4f} nm', file=logfile)

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

	plt.savefig(f'{plots_dir}/spectra_main.jpg')

	
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
		tracking_spectra.append(absorbs)

		if type(twl) is tuple: twlb = (twl[0]+twl[1])/2
		else: twlb = twl

		plt.scatter(np.ones_like(absorbs) * twlb, absorbs)

	plt.figure()
	plt.title('Tracked spectrax')
	plt.xlabel('t (s)')
	plt.ylabel('Absorbance (a.u.)')
	for twl, tabs in zip(tracking_spectrax, tracking_spectra):
		low = twl[0]
		high = twl[1]
		
		plt.plot(np.arange(N_spectra)*time_delay, tabs, label=r'$\lambda_{avg}$' + f' = [{low}, {high}] nm')

	plt.legend()
	plt.savefig(f'{plots_dir}/tracked_spectra.jpg')


	for twl, tabs in zip(tracking_spectrax, tracking_spectra):
		low = twl[0]
		high = twl[1]

		fit_exp_decay(np.arange(N_spectra)*time_delay, np.asarray(tabs), title=f'Predicted and reference rate (around $\lambda ∈ [{low}, {high}]$ nm)', plots_dir=plots_dir, use_scipy=True)


	# plt.show()	

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
	}


	cy = main(**settings['SP_cyclohexanone_UVVIS_10s_360x_20um'])[0]
	uncy = main(**settings['SP_cyclohexanone_UVVIS_10s_360x_20um_decoupled'])[0]
	cx = np.arange(cy.size)*10
	uncx = np.arange(uncy.size)*10

	plt.close('all')
	plt.figure()
	plt.title(r'Kinetics of average signal between 520 nm and 580 nm')
	plt.xlabel('t (s)')
	plt.ylabel('Absorbance (a.u.)')
	plt.plot(cx, cy, label='Coupled cyclohexanone')
	plt.plot(uncx, uncy, label='Uncoupled cyclohexanone')
	plt.legend()

	plt.figure()
	plt.title(r'Normalised kinetics of average signal between 520 nm and 580 nm')
	plt.xlabel('t (s)')
	plt.ylabel('Absorbance (normalised)')
	plt.plot(cx, (cy-cy.min())/(cy.max()-cy.min()), label='Coupled cyclohexanone')
	plt.plot(uncx, (uncy-uncy.min())/(uncy.max()-uncy.min()), label='Uncoupled cyclohexanone')
	
	coupled_fit = fit_exp_decay(cx, cy - np.mean(cy[-200:-1]), show_fit=True)
	uncoupled_fit = fit_exp_decay(uncx, uncy - np.mean(uncy[-200:-1]), show_fit=True)

	print('Coupled exponential fitting:')
	print(f'\tk  				= {coupled_fit["k"]:.3f} s^-1')
	print(f'\tA0 				= {coupled_fit["A0"]:.3f}')
	print(f'\tB  				= {coupled_fit["B"]:.3f}')
	print(f'\tMeant life-time   = {1/coupled_fit["k"]:.2f} s')
	print(f'\tHalf-life         = {np.log(2)/coupled_fit["k"]:.2f} s')

	print('Uncoupled exponential fitting:')
	print(f'\tk  				= {uncoupled_fit["k"]:.3f} s^-1')
	print(f'\tA0 				= {uncoupled_fit["A0"]:.3f}')
	print(f'\tB  				= {uncoupled_fit["B"]:.3f}')
	print(f'\tMeant life-time   = {1/uncoupled_fit["k"]:.2f} s')
	print(f'\tHalf-life         = {np.log(2)/uncoupled_fit["k"]:.2f} s')

	plt.show()