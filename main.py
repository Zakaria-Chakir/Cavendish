import cv2 as cv
import numpy as np
import scipy
from dot_detection import find_contour_mnk
import matplotlib.pyplot as plt
import csv
import ast
import webbrowser
import time as clock
import PIL as pillow
from Oscillation_modelling import *

start_time = clock.time()

########### Video tracking function ###############
total_time, found_circles = find_contour_mnk(video_input_path = r'Input_videos\eight_vid.mp4', video_output_path = r"Output_videos\output_arrrrrr.avi", 
                 pos_circle_txt_path = r"Position of circles\test_8_v3.txt", Countour_finding_param = {
                            'dp': 1,                   # Inverse of precision (lower = higher precision)
                            'mindist': 1000,             # Minimum distance between circle centers (px)
                            'canny_threshold': 4,      # Higher = fewer edges (but more precise)
                            'circle_threshold':5,     # Higher = stricter circle detection
                            'min_radius': 1,            # Minimum radius (px)
                            'max_radius': 20,            # Maximum radius (px)
                            }, scale_factor = 0.25, lower_color_bdd = np.array([40, 100, 40]), 
                            upper_color_bdd = np.array([240, 255, 240]), morb_kernel_size = (5,5), real_time_video = False,
                            mouse_curser = True, crop_xmin = 0, crop_xmax = -1, crop_ymin = 247, crop_ymax = 1503)
                            
                        


# scale_factor = 0.25, lower_color_bdd = np.array([65, 130, 75]), 
#                             upper_color_bdd = np.array([255, 255, 255]), morb_kernel_size = (4,4), real_time_video = True,
#                             mouse_curser = True, crop_xmin = 130, crop_xmax = 210, crop_ymin = 360, crop_ymax = 950)

########### Importing data from a textfile ##############
with open(r"Position of circles\test_8_v3.txt", 'r') as file:
    found_circles = [ast.literal_eval(line.strip()) for line in file]
    total_time = [i/30 for i in range(len(found_circles))] # sixth total_frames : 196157 seventh total_frames : 243569
found_circles = np.asarray(found_circles)


######## Visualising data and finding the period ##########
time, planar_disp, displacement_err = [], [], []
temp_t, temp_x, temp_err = [], [], []
i = 0
while i < len(found_circles):
    if found_circles[i][0] == None :
        i+=1
        continue
    temp_x.append(found_circles[i,0])
    temp_t.append(total_time[i])
    temp_err.append(found_circles[i,2])

    if i%300 == 0:
        
        x_mean, x_err = weighted_mean(temp_x, temp_err)
        t_mean = np.mean(temp_t)
        temp_t, temp_x, temp_err = [], [], []
        time.append(t_mean)
        planar_disp.append(x_mean)
        displacement_err.append(x_err)
    i+= 1




avg = np.mean([8*0.187*(10**-3)*x for x in planar_disp[0:]])
##### Curve fit #####
estimate_dict = {
    'C' : [0.005,2,0.2],
    'Z' : [0.1, 10, 1],
    'I': [5*10**-8, 10**-2, 3*10**-4],
    'damp' : [10**-12, 10**-7, 1e-8],
    'G' : [6.4*10**-14, 10**-8, 6.67e-11],
    'kappa': [1e-8, 5e-8, 3e-8],
    'phi' : [-np.pi, np.pi, 0],
    'k' : [-1, 1, 0.4],
}


Bounds_min = [x[0] for x in estimate_dict.values()]
Bounds_max = [x[1] for x in estimate_dict.values()]
Bounds = [Bounds_min, Bounds_max]
Initial_guess = [x[2] for x in estimate_dict.values()]
coeff, coeff_err = curve_fitting(time[0:], [8*0.187*(10**-3)*x-avg for x in planar_disp[0:]] , y_err = [8*0.187*(10**-3)*y for y in displacement_err[0:]], x_title = 'Time (s)', 
                                 y_title = 'Planar displacement (m)', model = torsional_oscillator_model,bdds = Bounds, 
                                 initial_guess = Initial_guess) # see function in oscillation model


# Print the parameters
for c,e, name, minimum, maximum in zip(coeff, coeff_err, ['C', 'Z', 'G', 'k'], 
            Bounds[0], Bounds[1]) :
    print(f"{name} : {c:.4e} +/- {e:.4e}. min: {minimum:.4e} max: {maximum:.4e}")


print()
print("--- %s seconds ---" % (clock.time() - start_time))

print()

#### Depricated code #####
# ['C', 'Z', 'I' ,'damp', 'G', 'kappa', 'phi', 'k', 'h', 'd', 'm_big', 'dist_mass', 'm_small']
# m_big = 1.5
# dist_mass = 3.7256e-02
# m_small = 38.3e-3
# d = 50e-3
# I = 2.3684e-04
# damp = 1.2587e-07
# kappa = 3.7617e-08
# C = 1.2270e-01
# G = 6.7854e-11 # 6.6169e-11
# # k = 2.9828e+02
# phi = -3.4453e+00


# b = damp/(2*I) 
# kappa_eff = kappa + 2*G*d*((m_small*m_big)/(dist_mass**2))
# w0 = np.sqrt(kappa_eff/I)
# w1 = np.sqrt(w0**2 - b**2)
# print(w1/(2*np.pi) , "Hz")













avg_angl = np.mean([np.arctan((8*0.187*(10**-3)*x-avg)/6.3089e-01) for x in planar_disp[15:-25] ])

def actual_angular_model(x, q, C):
    # All varibles : (x, C, Z, I, damp, G, kappa, phi, k, h, d, m_big, dist_mass, m_small)
    # Known values (might be subject to changes)
    m_big = 1.5
    dist_mass = 3.7256e-02
    m_small = 38.3e-3
    d = 50e-3
    I = 2.3714e-04 # 2.3684e-04
    damp = 9.5630e-08 # 1.2587e-07
    kappa = 3.5840e-08 # 3.7617e-08

    G = 4.6282e-10 # 6.6169e-11

    # k = 2.9828e+02
    phi = -2.9825e-01 # -3.4453e+00

    # C = 6.2798e-01/3 # 79.83e-03
    # # Z = 7.7772e+01
    # q = 0.02714468

    # Actual model
    b = damp/(2*I) 
    kappa_eff = kappa + 2*G*d*((m_small*m_big)/(dist_mass**2))
    w0 = np.sqrt(kappa_eff/I)
    w1 = np.sqrt(w0**2 - b**2)
    displacement = [C*np.e**(-b*(t))* np.cos(w1*(t)+phi) + q for t in x]
    return displacement





def plotting(x, y, y_err, x_title, y_title, 
                                ecolors = ["#FFB500", "#00A86B", "#0F5D8A", "#032A3A", "#D11A58"], 
                                colors = ["#FFD166", "#06D6A0", "#118AB2", "#073B4C", "#EF476F"], ures = '(rad)'):
    
    def linear_regression_with_errors(x, y, y_err):
        popt, pcov = curve_fit(actual_angular_model, xdata = x, ydata = y, sigma=y_err, absolute_sigma=True, bounds = [[-0.04, 0.05], [0.04, 1]], p0 = [0,0.6], ftol = 2.22e-15)
        return popt, np.sqrt(np.diag(pcov))
    
    fig, (ax, ax_res) = plt.subplots(nrows=2, ncols=1, figsize=(10,7), gridspec_kw={'height_ratios': [3, 1]})

    x_model = np.linspace(np.min(x), 
                        np.max(x), 40000)
    param, _ = linear_regression_with_errors(x, y, y_err)
    ax.errorbar(x, y, yerr=y_err, fmt='o', capsize=6, capthick=1, color='#657ca2', ecolor='#8c89de', elinewidth=1, 
                    markersize=4, alpha  = 0.8)
    


    # ax.plot(x_model, 100*np.cos(2*np.pi*0.0019877*x_model-7.8540e-01)+5.4611e+02)
    ax.plot(x_model, actual_angular_model(x_model, *param), linestyle='-', linewidth=1, alpha = 0.6, color = '#2D004B')
    ax_res.set_xlabel(x_title, fontsize = 14)
    ax.set_ylabel(y_title, fontsize = 14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    res = [y_i-mod_i for y_i, mod_i in zip(y,actual_angular_model(x, *param))]
    
    print(len(y), len(res), len(actual_angular_model(x, *param)))
    ax_res.errorbar(x, res, yerr=y_err, color='#FF6B6B', fmt='o', capsize=6, capthick=1, 
                    ecolor=ecolors[4], elinewidth=1, markersize=4)
    ax_res.axhline(color='black', linestyle='--')
    ax_res.set_ylabel(f'Residuals {ures}', fontsize = 14)
    ax_res.spines['top'].set_visible(False)
    ax_res.spines['right'].set_visible(False) 
    

    print('The residuals standard deviation is :', np.std(res, ddof=1))

    # for i in range(len(coeff)):
    #     print(f'{coeff[i] } +/- {coeff_error[i]}')

    plt.show()

    print('Chi^2 is :', chi2(x, y, param, y_err, model = actual_angular_model), f'for {len(x)-len(coeff)} degrees of freedom')


plotting(time[15:-25], [(1/4)*np.arctan((8*0.187*(10**-3)*x-avg)/6.3089e-01) for x in planar_disp[15:-25] ], 
         y_err = [(8*(10**-3)*y)/(1+((8*0.187*(10**-3)*x-avg))**2) for x,y in zip(planar_disp[15:-25], displacement_err[15:-25])],
         x_title = r'Time, $t$ (s)', y_title = r'Angular displacement, $\theta$ (rad)')




























# # ax.plot(x_model, 100*np.cos(2*np.pi*0.0019877*x_model-7.8540e-01)+5.4611e+02)
# # ax.plot(x_model, model(x_model, *coeff), linestyle='-', linewidth=1, alpha = 0.6, color = 'green')
# # ax.set_xlabel(x_title, fontsize = 14)
# # ax.set_ylabel(y_title, fontsize = 14)
# # ax.spines['top'].set_visible(False)
# # ax.spines['right'].set_visible(False)

# # res = [y_i-mod_i for y_i, mod_i in zip(y,model(x, *coeff))]
# # res_coeff, coeff_error = linear_regression_with_errors(x, y, y_err, model)
# # ax_res.errorbar(x, res, yerr=y_err, color=colors[4], fmt='o', capsize=6, capthick=1, 
# #                 ecolor=ecolors[4], elinewidth=1, markersize=4)
# # ax_res.axhline(color='black', linestyle='--')
# # ax_res.set_ylabel(f'Residuals {ures}', fontsize = 14)
# # ax_res.spines['top'].set_visible(False)
# # ax_res.spines['right'].set_visible(False) 


# ########################################################################
# # [15:-25]
# # i%600
# # estimate_dict = {
# #     'C' : [0.005,2,0.2],
# #     'Z' : [0.1, 4, 1],
# #     # 'I': [5*10**-8, 10**-2, 3*10**-4],
# #     # 'damp' : [10**-12, 10**-4, 1e-9],
# #     'G' : [6.4*10**-14, 10**-9, 6.67e-11],
# #     # 'kappa': [1e-12, 1e-3, 3e-8],
# #     # 'phi' : [-np.pi, np.pi, 0],
# #     'k' : [-1, 1, 0.4],
# #     # 'd' : [0.003, 0.7, 0.05],
# #     # 'm_big' : [1.48, 1.52, 1.5],
# #     # 'dist_mass' : [35*10**-3, 44*10**-3, 39*10**-3],
# #     # 'm_small' : [0.0020,0.8, 0.04]
    

# # }

# # # Known values (might be subject to changes)
# #     m_big = 1.5
# #     dist_mass = 3.7256e-02
# #     m_small = 38.3e-3
# #     d = 50e-3
# #     I = 2.3684e-04
# #     damp = 1.2587e-07
# #     kappa = 3.7617e-08

# #     # k = 2.9828e+02
# #     # phi = -3.4453e+00

# #     # C = 79.83e-03
# #     # Z = 7.7772e+01
