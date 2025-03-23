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
found_circles = find_contour_mnk(video_input_path = 'Input_videos\Second_video_data_v1.mp4', video_output_path = "Output_videos\Second_test_tracking.avi", 
                 pos_circle_txt_path = "Position of circles\Second_test_position_of_dot.txt", Countour_finding_param = {
                            'dp': 1,                   # Inverse of precision (lower = higher precision)
                            'mindist': 6,             # Minimum distance between circle centers (px)
                            'canny_threshold': 6,      # Higher = fewer edges (but more precise)
                            'circle_threshold': 6,     # Higher = stricter circle detection
                            'min_radius': 1,            # Minimum radius (px)
                            'max_radius': 5,            # Maximum radius (px)
                            }, scale_factor = 0.25, lower_color_bdd = np.array([30, 100, 30]), 
                            upper_color_bdd = np.array([180, 255, 200]), morb_kernel_size = (4,4), real_time_video = True,
                            mouse_curser = False)



########### Importing data from a textfile ##############
with open("Position of circles\Second_test_position_of_dot.txt", 'r') as file:
    found_circles = [ast.literal_eval(line.strip()) for line in file]
found_circles = np.asarray(found_circles)


######## Visualising data and finding the period ##########
time, planar_disp, displacement_err = [], [], []
temp_t, temp_x, temp_err = [], [], []

i = 0
while i < len(found_circles):
    temp_x.append(found_circles[i,0])
    temp_t.append(i/30)
    temp_err.append(found_circles[i,2])

    if i%400 == 0:
        
        x_mean, x_err = weighted_mean(temp_x, temp_err)
        t_mean = np.mean(temp_t)
        temp_t, temp_x, temp_err = [], [], []
        time.append(t_mean)
        planar_disp.append(x_mean)
        displacement_err.append(x_err*5)
    i+= 1

# plt.plot(time, planar_disp)
# plt.show()

fft_signal = np.abs(scipy.fft.rfft(planar_disp))
fft_freq = scipy.fft.rfftfreq(len(time), d=1/60)


idx_max, _ = scipy.signal.find_peaks(fft_signal, threshold = 10000, distance = 60, prominence = 100)
# print(fft_freq[idx_max])
# plt.plot(fft_freq, fft_signal, color = 'gray')
# plt.scatter(fft_freq[idx_max], fft_signal[idx_max], color = 'orange', marker = 'x')
# plt.show()



##### Curve fit #####
coeff, coeff_err = curve_fitting(time[1:], planar_disp[1:], y_err = displacement_err[1:], 
                                 x_title = 'Time (s)', y_title = 'Planar displacement (px)', 
                                 model = torsional_oscillator_model,
                                 bdds = [[1, 10**-5, 0, 0, 0, 0, 7*10**-7], [100, 10, 10, 10**-5, 2*np.pi, 400, 10**-5]],
                                initial_guess = [50, 10**-3,0.0000001, 7*10**-11, np.pi, 150, 4*10**-6])
for c,e, name, maximum, minimum in zip(coeff, coeff_err, ['C', 'I' ,'damp', 'G', 'phi', 'k', 'kappa'], 
            [1, 10**-5, 0, 0, 0, 0, 10**-7], [100, 10, 10, 10**-5, 2*np.pi, 400, 10**-5]) :
    print(f"{name} : {c:.4e} +/- {e:.4e}. Max : {maximum:.4e}     . Min : {minimum:.4e}  ")


print()
print("--- %s seconds ---" % (clock.time() - start_time))


