import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import pkg.peak_finding as pf




def analyse_kinetics(file, dt):
	if file.endswith('.txt'):
		data = np.genfromtxt(file, skip_header=1)
	elif file.endswith('.csv'):
		with open(file, 'r') as f:
			content = f.readlines()
		if any(';' in l for l in content):
			content = [l.replace(',', '.').replace(';', ',') for l in content]
		data = np.genfromtxt(content, skip_header=1, delimiter=',')[:,:-1]


	spectrax = data[:,0]
	spectray = data[:,1:]

	t = np.arange(spectray.shape[1]) * dt
	avg = spectray.mean(axis=0)

	f = lambda x, *args: args[0]*np.exp(-1/args[2]*x) + args[1]*np.exp(-x/args[3])
	popt, pcov = curve_fit(f, t, avg, p0=[0.7,0.1,200,2000])

	return {'x': t, 'y': avg, 'fit': f(t, *popt), 't1':popt[2]}


def analyse_cavity(file):
	with open(file, 'r') as f:
		content = f.readlines()
		if any(';' in l for l in content):
			content = [l.replace(',', '.').replace(';', ',') for l in content]

	data = np.genfromtxt(content, skip_header=2, delimiter=',')[:,:-1]
	x = data[:,0]
	y = data[:,1]

	peaks = pf.get_peaks(x, y, prominence=0.05)
	FSR = pf.get_FSR(peaks['peakx'])
	return {'FSR':FSR['FSR'], 'offset':FSR['offset'], 'v':sorted([FSR['FSR']*i for i in range(25)], key=lambda f: abs(f-1350))[0]}



if __name__ == '__main__':

	cmap = plt.get_cmap('tab10')

	dts = [20, 20, 10, 20]
	kinetic_files = ['20211207_shim_1/all.txt', '20211208_shim_2/all.txt', '20211208_HP_2/kinetics.csv', '20211208_shim_3/all.txt']
	kinetics_results = [analyse_kinetics(f, dt) for f, dt in zip(kinetic_files, dts)]
	cavity_files = ['20211207_shim_1/cavity.csv', '20211208_shim_2/cavity.csv', '20211208_HP_2/cavity.csv', '20211208_shim_3/cavity.csv']
	cavity_results = [analyse_cavity(f) for f in cavity_files]
	tuned_wns = [c['v'] for c in cavity_results]
	k1s = [1000/k['t1'] for k in kinetics_results]

	print('Summary')
	print('Exp. | v (cm^-1) | k1 (s^-1) | t1 (s)')
	for i, t, k in zip(range(len(tuned_wns)), tuned_wns, k1s):
		print(f'{i: <4} | {t:9.2f} | {k/1000:9.3E} | {1000/k:.2f}')

	plt.subplot(1,2,1)
	plt.gca().set_title('Kinetic profile of experiments')
	for i, k in enumerate(kinetics_results):
		plt.scatter(k['x'], k['y'], label=f'Exp. {i}', color=cmap(i))
		plt.plot(k['x'], k['fit'], color=cmap(i))
	plt.legend()
	plt.xlabel('t (s)')
	plt.ylabel('Absorption (a.u.)')



	ax = plt.subplot(1,2,2)

	plt.gca().set_title('Rate constants and cyclohexanone FTIR spectrum')

	chxo_data = np.genfromtxt('cyclohexanone_3um.csv', skip_header=2, delimiter=',')
	chxo_x = chxo_data[:,0]
	chxo_y = chxo_data[:,1]

	ax2 = plt.twinx(ax)

	ax.plot(chxo_x, chxo_y, label='IR spectrum cyclohexanone',color='black')
	ax.set_xlim(min(tuned_wns)-30, max(tuned_wns)+30)
	ax.set_ylim(0, 1.25)
	ax.set_xlabel(r'$\nu (cm^{-1})$')
	ax.set_ylabel('Transmission')
	# print(tuned_wns, k1s)
	for i, t, k in zip(range(len(tuned_wns)), tuned_wns, k1s):
		ax2.scatter(t, k, color=cmap(i))
	ax2.spines['right'].set_color('red')
	ax2.yaxis.label.set_color('red')
	ax2.tick_params(axis='y', colors='red')
	ax.set_xlim(min(tuned_wns)-30, max(tuned_wns)+30)
	ax2.set_ylim(0, max(k1s)*1.2)
	ax2.set_ylabel(rf'Reaction rate (k1) $(10^{-3}s^{-1})$')

	plt.tight_layout()
	plt.show()