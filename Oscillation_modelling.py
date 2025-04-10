import numpy as np
import scipy 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution



def torsional_oscillator_model(x, C, Z, G, k):
    # All varibles : (x, C, Z, I, damp, G, kappa, phi, k, h, d, m_big, dist_mass, m_small)
    # Known values (might be subject to changes)
    m_big = 1.5
    dist_mass = 3.7256e-02
    m_small = 38.3e-3
    d = 50e-3
    I = 2.3684e-04
    damp = 1.2587e-07
    kappa = 3.7617e-08

    # k = 2.9828e+02
    phi = -3.4453e+00

    # C = 79.83e-03
    # Z = 7.7772e+01

    # Actual model
    b = damp/(2*I) 
    kappa_eff = kappa + 2*G*d*((m_small*m_big)/(dist_mass**2))
    w0 = np.sqrt(kappa_eff/I)
    w1 = np.sqrt(w0**2 - b**2)
    theta = [Z*np.tan(C*np.e**(-b*(t))* np.cos(w1*(t)+phi))+ k for t in x]
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

def weighted_mean(val, err):
    values = np.array(val)
    errors = np.array(err)

    weights = 1 / (errors**2)
    weighted_mean = np.sum(weights * values) / np.sum(weights)
    weighted_mean_error = np.sqrt(1 / np.sum(weights))
    
    return weighted_mean, weighted_mean_error




def curve_fitting(x, y, y_err, x_title, y_title, model, linecolors = ["#FF9E00", "#00965B", "#0A4A70", "#021D27", "#EF476F"], 
                                ecolors = ["#FFB500", "#00A86B", "#0F5D8A", "#032A3A", "#D11A58"], 
                                colors = ["#FFD166", "#06D6A0", "#118AB2", "#073B4C", "#EF476F"], ures = '(m)',
                                bdds = [[1, 10**-5, 0, 0, 0, 0, 10**-7], [100, 10, 10, 10**-5, 2*np.pi, 400, 10**-4]],
                                initial_guess = [50, 10**-3,0.0000001, 7*10**-10, 0.01, 150, 4*10**-6]):
    """
    Performs curve fitting using non-linear least squares regression, plots the 
    fitted curve along with residuals, and computes statistical metrics.

    Parameters:
    ----------
    x : array-like
        The independent variable (data points).
    y : array-like
        The dependent variable (measured values).
    y_err : array-like
        The uncertainties (errors) in y.
    x_title : str
        Label for the x-axis.
    y_title : str
        Label for the y-axis.
    model : function
        The mathematical function to fit to the data.
    linecolors : list of str, optional
        Colors for the fitted lines (default provided).
    ecolors : list of str, optional
        Colors for error bars (default provided).
    colors : list of str, optional
        Colors for data points (default provided).
    ures : str, optional
        Units for the residuals axis label (default: '(rad)').
    bdds : list of lists, optional
        Bounds for curve fitting as [lower_bounds, upper_bounds].
    initial_guess : list, optional
        Initial parameter estimates for curve fitting.

    Returns:
    -------
    coeff : array
        The best-fit parameters for the model.
    coeff_error : array
        The standard errors of the fitted parameters.

    Additional Outputs:
    -------------------
    - Displays the fitted curve and residuals plot.
    - Prints the residuals' standard deviation.
    - Computes and prints the chi-squared statistic.
    """
    def linear_regression_with_errors(x, y, y_err, model):
        popt, pcov = curve_fit(model, xdata = x, ydata = y, sigma=y_err, absolute_sigma=True, bounds = bdds, maxfev = 100000, p0 = initial_guess, ftol = 2.22e-15)
        return popt, np.sqrt(np.diag(pcov))
    
    fig, (ax, ax_res) = plt.subplots(nrows=2, ncols=1, figsize=(10,7), gridspec_kw={'height_ratios': [3, 1]})

    x_model = np.linspace(np.min(x), 
                        np.max(x), 40000)
    coeff, coeff_error = linear_regression_with_errors(x, y, y_err, model)
    ax.errorbar(x, y, yerr=y_err, fmt='o', capsize=6, capthick=1, color=colors[2], ecolor=ecolors[2], elinewidth=1, 
                    markersize=4)
    # ax.plot(x_model, 100*np.cos(2*np.pi*0.0019877*x_model-7.8540e-01)+5.4611e+02)
    ax.plot(x_model, model(x_model, *coeff), linestyle='-', linewidth=1, alpha = 0.6, color = 'green')
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

    # for i in range(len(coeff)):
    #     print(f'{coeff[i] } +/- {coeff_error[i]}')

    plt.show()

    print('Chi^2 is :', chi2(x, y, coeff, y_err, model = model), f'for {len(x)-len(coeff)} degrees of freedom')
    return coeff, coeff_error


