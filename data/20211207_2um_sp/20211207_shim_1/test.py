import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def get_data(file):

	data = np.genfromtxt('kinetics.csv', skip_header=2, delimiter=',')[:,:-1]



	spectrax = data[:,0]
	spectray = data[:,1:]

	dt = 20 #in s
	t = np.arange(spectray.shape[1]) * dt
	avg = spectray.mean(axis=0)
	avg = avg[np.logical_and(t!=680, t!=740)]
	t = np.arange(avg.size) * dt

	f = lambda x, *args: args[0]*np.exp(-1/args[2]*x) + args[1]*np.exp(-x/args[3])
	popt, pcov = curve_fit(f, t, avg, p0=[0.7,0.1,200,2000])
	# print('fit complete')
	# print(f'lifetime1 = {popt[2]:.2f} s')
	# print(f'lifetime2 = {popt[3]:.2f} s')

	# plt.subplot(1,2,1)
	# plt.suptitle('Kinetics measurement on Shimadzu with 2um spacer tuned to 1350cm^-1')
	# plt.title('Time resolved spectrum')
	# plt.xlabel('t (s)')
	# plt.ylabel('Wavelength (nm)')
	# plt.imshow(spectray, aspect='auto', origin='bottom', extent=[t.min(), t.max(), spectrax.min(), spectrax.max()])
	# plt.subplot(1,2,2)
	# plt.title(f'Kinetics (t1={popt[2]:.2f}s, t2={popt[3]:.2f}s)')
	# plt.xlabel('t (s)')
	# plt.ylabel('Average absorbance (a.u.)')
	# plt.scatter(t, avg, color='red', label='Exp.')
	# plt.plot(t, f(t, *popt), color='blue', label='Model')
	# plt.legend()
	# plt.show()