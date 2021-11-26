import numpy as np 
import matplotlib.pyplot as plt
import scipy.signal
from pkg.peak_finding import get_peaks


def read_spectra(file):
	with open(file, 'r') as f:
		lines = f.readlines()
		spectrum_angles = lines[0].strip().split(',')[1:]
		spectrum_angles = [float(a) for a in spectrum_angles if a != '']
		spectrum_types = lines[1].lower().strip().strip(',').split(',')[1:]
	data = np.genfromtxt(file, skip_header=2, delimiter=',')[:,:-1]  #dont send last row cause itis empty
	return data, np.asarray(spectrum_angles), spectrum_types


def main(file):
	low_wn = 1500
	high_wn = 1900


	spectra, angles, types = read_spectra(file)

	angles = np.abs(angles/100 - 180) * 30.5/20
	angleidx = np.argsort(angles) #sort the spctra based on angle

	spectrax = spectra[:, 0]
	dx = abs(np.mean(np.diff(spectrax)))
	spectray = spectra[:, 1:].T
	spectray = spectray[angleidx]
	angles = angles[angleidx]
	cmap = plt.get_cmap('jet')
	# peaksx = []
	for i, s in enumerate(spectray):
		# peakres = get_peaks(spectrax, s, prominence=0.005)
		# peaksx.append(peakres['peakx'])
		plt.plot(spectrax, s, color=cmap(i/angles.size))
	# peaksx = np.asarray(peaksx)

	# plt.plot(spectrax, spectray)

	#cut out desired wns
	lowidx = np.argmin(abs(spectrax - low_wn))
	highidx = np.argmin(abs(spectrax - high_wn))

	#normalize them 
	cut_spec = spectray[:, highidx:lowidx]
	cut_spec = cut_spec - cut_spec.min(axis=0)
	norm_spec = cut_spec/cut_spec.max(axis=0)
	norm_spec = cut_spec

	#copy and mirror the spectra
	copy = norm_spec[::-1].copy()
	norm_spec = np.vstack((copy[:-1], norm_spec))

	plt.figure()
	plt.ylabel('Wavenumber ($cm^{-1}$)')
	plt.xlabel('Angle (deg)')

	plt.imshow(norm_spec.T, extent=(max(angles), -max(angles), low_wn, high_wn), aspect='auto', cmap='jet')
	for s, a in zip(norm_spec, angles):
		print('hellow')
		peaks, _ = scipy.signal.find_peaks(s, prominence=0.0005)
		# print(peaks*dx+low_wn)
		peak_pos = []
		for peak in peaks:
			peak_pos.append((high_wn-peak*dx, max(angles)-a))
			peak_pos.append((high_wn-peak*dx, -max(angles)+a))

		# for x, y in peak_pos:
		# 	c = plt.get_cmap('rainbow')((x-low_wn)/(high_wn-low_wn))
		# 	plt.scatter(y, x, color=c)


	plt.figure()
	kparallel = 2*np.pi*np.sin(angles*np.pi/180) * 1720/10000 # in um^-1
	copy = kparallel[::-1].copy()
	kparallel = np.hstack((-copy[:-1], kparallel))

	# copy = peaksx[::-1].copy()
	# peaksx = np.hstack((-copy[:-1], peaksx))

	lowE = low_wn/10000 * 1.2398 #in eV
	highE = high_wn/10000* 1.2398 #in eV
	x, y = np.meshgrid(kparallel, np.linspace(highE, lowE, norm_spec.shape[1]))
	plt.pcolormesh(x, y, norm_spec.T, cmap='jet', shading='gouraud')
	plt.title('Spectral evolution with changing beam angle')
	plt.plot((kparallel.min(), kparallel.max()), (1720/10000* 1.2398,1720/10000* 1.2398), linestyle='dashed', color='black')
	plt.ylabel('E (eV)')
	plt.xlabel(r'$k_\parallel$ ($\mu m^{-1}$)')
	plt.show()


if __name__ == '__main__':
	main("data/20211117_cyclohexane_angle_dependence_3/data.csv")
