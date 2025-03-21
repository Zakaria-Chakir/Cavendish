import numpy as np
import scipy 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def torsional_oscillator_model(x, C, damp, I, kappa, G, phi, m_small, dist_mass, d, k):
    # Known values (might be subject to changes)
    m_big = 1.5

    # Actual model
    b = 2*damp/I 
    kappa_eff = kappa + 2*G*d*((m_small*m_big)/(dist_mass))
    w0 = np.sqrt(kappa_eff/I)
    w1 = np.sqrt(w0**2 - b**2)
    theta = [C*np.e**(-b*t)* np.cos(w1*t+phi) + k for t in x]
    return theta 

def rnd_color(n):
    colors = []
    for _ in range(n):
        color = np.random.uniform(0.2, 0.7, 3)
        colors.append(tuple(color))
    return colors


def chi2(x, y, para, err, model):
    y = np.array(y)
    mod_y = np.array(model(x,*para))
    err = np.array(err)
    chi2 = np.sum(((y - mod_y) / err) ** 2)
    return chi2


def curve_fitting(x, y, y_err, x_title, y_title, model, linecolors = ["#FF9E00", "#00965B", "#0A4A70", "#021D27", "#EF476F"], 
                                ecolors = ["#FFB500", "#00A86B", "#0F5D8A", "#032A3A", "#D11A58"], 
                                colors = ["#FFD166", "#06D6A0", "#118AB2", "#073B4C", "#EF476F"], ures = 'rad'):
    
    def linear_regression_with_errors(x, y, y_err, model):
        popt, pcov = curve_fit(model, xdata = x, ydata = y, sigma=y_err, absolute_sigma=True, bounds = [[20, 0, 0, 0, 0, 0, 0, 0, 0, 0], [50, 0.000001, 10, np.inf, 0.001, 6.283185307179586, 0.0025, 0.05, 0.1, 400]], maxfev = 10000, p0 = [30, 0.00000000000001, 1, 1, 6*10**-9, 0, 0.0015, 0.01, 0.05, 150])
        return popt, np.sqrt(np.diag(pcov))
    
    fig, (ax, ax_res) = plt.subplots(nrows=2, ncols=1, figsize=(10,7), gridspec_kw={'height_ratios': [3, 1]})

    x_model = np.linspace(np.min(x), 
                        np.max(x), 40000)
    coeff, coeff_error = linear_regression_with_errors(x, y, y_err, model)
    ax.errorbar(x, y, yerr=y_err, fmt='o', capsize=6, capthick=1, color=colors[2], ecolor=ecolors[2], elinewidth=1, 
                    markersize=4)
    ax.plot(x_model, model(x_model, *coeff), color=linecolors[2], linestyle='-', linewidth=1, alpha = 0.6)
    ax.set_xlabel(x_title, fontsize = 14)
    ax.set_ylabel(y_title, fontsize = 14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    res = [y_i-mod_i for y_i, mod_i in zip(y,model(x, *coeff))]
    res_coeff, coeff_error = linear_regression_with_errors(x, y, y_err, model)
    ax_res.errorbar(x, res, yerr=y_err, color=colors[4], fmt='o', capsize=6, capthick=1, 
                    ecolor=ecolors[4], elinewidth=1, markersize=4)
    ax_res.axhline(color='black', linestyle='--')
    ax_res.set_ylabel(f'Residuals {ures}', fontsize = 14)
    ax_res.spines['top'].set_visible(False)
    ax_res.spines['right'].set_visible(False) 
    

    print('The residuals standard deviation is :', np.std(res, ddof=1))

    for i in range(len(coeff)):
        print(f'{coeff[i] } +/- {coeff_error[i]}')

    plt.show()

    print('Chi^2 is :', chi2(x, y, coeff, y_err, model = model), f'for {len(x)-len(coeff)} degrees of freedom')
    return coeff, coeff_error


