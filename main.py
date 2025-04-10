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
# total_time, found_circles = find_contour_mnk(video_input_path = r'Input_videos\20250407_142856.mp4', video_output_path = r"Output_videos\seventh_vid_out.avi", 
#                  pos_circle_txt_path = r"Position of circles\test_7.txt", Countour_finding_param = {
#                             'dp': 1,                   # Inverse of precision (lower = higher precision)
#                             'mindist': 100,             # Minimum distance between circle centers (px)
#                             'canny_threshold': 8,      # Higher = fewer edges (but more precise)
#                             'circle_threshold':8,     # Higher = stricter circle detection
#                             'min_radius': 2,            # Minimum radius (px)
#                             'max_radius': 18,            # Maximum radius (px)
#                             }, scale_factor = 0.25, lower_color_bdd = np.array([65, 130, 75]), 
#                             upper_color_bdd = np.array([255, 255, 255]), morb_kernel_size = (4,4), real_time_video = False,
#                             mouse_curser = True, crop_xmin = 130, crop_xmax = 210, crop_ymin = 360, crop_ymax = 950)



########### Importing data from a textfile ##############
with open(r"Position of circles\test_7.txt", 'r') as file:
    found_circles = [ast.literal_eval(line.strip()) for line in file]
    total_time = [i/30 for i in range(243569)] # sixth total_frames : 196157
found_circles = np.asarray(found_circles)

# plt.scatter(found_circles[:,0], found_circles[:,1])
# plt.show()
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

    if i%600 == 0:
        
        x_mean, x_err = weighted_mean(temp_x, temp_err)
        t_mean = np.mean(temp_t)
        temp_t, temp_x, temp_err = [], [], []
        time.append(t_mean)
        planar_disp.append(x_mean)
        displacement_err.append(x_err)
    i+= 1

#### Fourier transform of the signal to find the period ######
# fft_signal = np.abs(scipy.fft.rfft(planar_disp))
# fft_freq = scipy.fft.rfftfreq(len(time), d=1/30*300) 


# idx_max, _ = scipy.signal.find_peaks(fft_signal, threshold = 100, distance = 60, prominence = 10) # this finds the peaks in fourier space
# print(fft_freq[idx_max])
# plt.plot(fft_freq, fft_signal, color = 'gray')
# plt.scatter(fft_freq[idx_max], fft_signal[idx_max], color = 'orange', marker = 'x')
# plt.title(f'Fourier transform')
# plt.show()

avg = np.mean([2.8*(10**-3)*x for x in planar_disp[15:-25]])
##### Curve fit #####
estimate_dict = {
    'C' : [0.005,2,0.2],
    'Z' : [0.1, 4, 1],
    # 'I': [5*10**-8, 10**-2, 3*10**-4],
    # 'damp' : [10**-12, 10**-4, 1e-9],
    'G' : [6.4*10**-14, 10**-9, 6.67e-11],
    # 'kappa': [1e-12, 1e-3, 3e-8],
    # 'phi' : [-np.pi, np.pi, 0],
    'k' : [-1, 1, 0.4],
    # 'd' : [0.003, 0.7, 0.05],
    # 'm_big' : [1.48, 1.52, 1.5],
    # 'dist_mass' : [35*10**-3, 44*10**-3, 39*10**-3],
    # 'm_small' : [0.0020,0.8, 0.04]
    

}


Bounds_min = [x[0] for x in estimate_dict.values()]
Bounds_max = [x[1] for x in estimate_dict.values()]
Bounds = [Bounds_min, Bounds_max]
Initial_guess = [x[2] for x in estimate_dict.values()]
coeff, coeff_err = curve_fitting(time[15:-25], [2.8*(10**-3)*x-avg for x in planar_disp[15:-25]] , y_err = [2.8*(10**-3)*y for y in displacement_err[15:-25]], x_title = 'Time (s)', 
                                 y_title = 'Planar displacement (m)', model = torsional_oscillator_model,bdds = Bounds, 
                                 initial_guess = Initial_guess) # see function in oscillation model


# Print the parameters
for c,e, name, minimum, maximum in zip(coeff, coeff_err, ['C', 'Z', 'G', 'k'], 
            Bounds[0], Bounds[1]) :
    print(f"{name} : {c:.4e} +/- {e:.4e}. min: {minimum:.4e} max: {maximum:.4e}")


print()
print("--- %s seconds ---" % (clock.time() - start_time))

print()
print(np.mean(planar_disp[140:-50]))

##### Depricated code #####
# ['C', 'Z', 'I' ,'damp', 'G', 'kappa', 'phi', 'k', 'h', 'd', 'm_big', 'dist_mass', 'm_small']
# b = 2*coeff[2]/coeff[1]
# kappa_eff = coeff[6] + 2*coeff[3]*coeff[8]*((coeff[-1]*coeff[-3])/(coeff[-2]))
# w0 = np.sqrt(kappa_eff/coeff[1])
# w1 = np.sqrt(w0**2 - b**2)
# print(w1/(2*np.pi) , "Hz")

