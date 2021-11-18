import numpy as np 
import matplotlib.pyplot as plt
import scipy.signal


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
	for i, s in enumerate(spectray):
		plt.plot(spectrax, s, color=cmap(i/angles.size))

	# plt.plot(spectrax, spectray)
	plt.show()


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

	plt.xlabel('Wavenumber (cm^-1)')
	plt.ylabel('Angle (deg)')

	plt.imshow(norm_spec, extent=(high_wn, low_wn, max(angles), -max(angles)), aspect='auto', cmap='jet')
	for s, a in zip(norm_spec, angles):
		peaks, _ = scipy.signal.find_peaks(s, prominence=0.0005)
		# print(peaks*dx+low_wn)
		for peak in peaks:
			plt.scatter(high_wn-peak*dx, max(angles)-a)
			plt.scatter(high_wn-peak*dx, -max(angles)+a)

	plt.show()



if __name__ == '__main__':
	main("data/20211117_cyclohexane_angle_dependence_3/data.csv")