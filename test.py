import numpy as np 
import matplotlib.pyplot as plt
from pkg.exp_decay_fitter import fit_exp_decay, fit_biexp_decay, fit_exp_lin


data = np.load("kinetics_cyclohexanone_20s_180x_5um_1_array.npy").flatten()
data = data - data.min()
x = np.arange(0, data.size) * 10

exp_lin 	= lambda x, *p: p[1]*np.exp(-p[0]*x) + p[2]*x
exp 		= lambda x, *p: p[1]*np.exp(-p[0]*x)
biexp 		= lambda x, *p: p[2]*np.exp(-p[0]*x) + p[3]*np.exp(-p[1]*x)


res_exp_lin = fit_exp_lin(x, data, show_fit=False, show_residuals=False)
res_exp 	= fit_exp_decay(x, data, show_fit=False, show_residuals=False)
res_biexp 	= fit_biexp_decay(x, data, show_fit=False, show_residuals=False)

# plt.plot(x, f(x, k, A0, b), label='input data')
normalize = lambda y: (y-y.min())/(y.max()-y.min())
normalize = lambda y: y

plt.figure()
plt.plot(x, data, label='Raw data')
plt.plot(x, exp(x, *res_exp.values()), label='Exponential model')
plt.plot(x, exp_lin(x, *res_exp_lin.values()), label='Exponential + linear model')
plt.plot(x, biexp(x, *res_biexp.values()), label='Biexponential model')
plt.legend()

plt.figure()
plt.plot(x, normalize(data - exp(x, *res_exp.values())), label='residuals of exp fit')
plt.plot(x, normalize(data - exp_lin(x, *res_exp_lin.values())), label='residuals of exp_lin fit')
plt.plot(x, normalize(data - biexp(x, *res_biexp.values())), label='residuals of biexp fit')
plt.legend()
plt.show()