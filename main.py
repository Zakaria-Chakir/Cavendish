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
# total_time, found_circles = find_contour_mnk(video_input_path = r'Input_videos\Second_video_data_v1.mp4', video_output_path = r"Output_videos\test.avi", 
#                  pos_circle_txt_path = r"Position of circles\test.txt", Countour_finding_param = {
#                             'dp': 1,                   # Inverse of precision (lower = higher precision)
#                             'mindist': 10,             # Minimum distance between circle centers (px)
#                             'canny_threshold': 6,      # Higher = fewer edges (but more precise)
#                             'circle_threshold': 6,     # Higher = stricter circle detection
#                             'min_radius': 1,            # Minimum radius (px)
#                             'max_radius': 7,            # Maximum radius (px)
#                             }, scale_factor = 0.25, lower_color_bdd = np.array([30, 100, 30]), 
#                             upper_color_bdd = np.array([240, 255, 250]), morb_kernel_size = (5,5), real_time_video = True,
#                             mouse_curser = False)



########### Importing data from a textfile ##############
with open("Position of circles\Second_test_position_of_dot.txt", 'r') as file:
    found_circles = [ast.literal_eval(line.strip()) for line in file]
    total_time = [i/30 for i in range(70796)]
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

# plt.plot(time, planar_disp)
# plt.show()
print(len(time), len(planar_disp))

fft_signal = np.abs(scipy.fft.rfft(planar_disp))
fft_freq = scipy.fft.rfftfreq(len(time), d=1/60)


idx_max, _ = scipy.signal.find_peaks(fft_signal, threshold = 10000, distance = 60, prominence = 100)
# print(fft_freq[idx_max])
# plt.plot(fft_freq, fft_signal, color = 'gray')
# plt.scatter(fft_freq[idx_max], fft_signal[idx_max], color = 'orange', marker = 'x')
# plt.show()



##### Curve fit #####
Bdds = [[10, 10**-6, 0, 0, -np.pi, 140, 8*10**-9, -1000], [100, 10**-2, 10**-2, 10**-5,np.pi, 200, 2*10**-5, 1000]]
coeff, coeff_err = curve_fitting(time[50:], planar_disp[50:] , y_err = displacement_err[50:], x_title = 'Time (s)', y_title = 'Angular displacement (rad)', model = torsional_oscillator_model,bdds = Bdds, initial_guess = [40, 10**-3,0.00000000001, 7*10**-11, 0, np.mean(planar_disp[50:]), 8*10**-8, 900])

for c,e, name, maximum, minimum in zip(coeff, coeff_err, ['C', 'I' ,'damp', 'G', 'phi', 'k', 'kappa','h'], 
            Bdds[0], Bdds[1]) :
    print(f"{name} : {c:.4e} +/- {e:.4e}. Max : {maximum:.4e}     . Min : {minimum:.4e}  ")


print()
print("--- %s seconds ---" % (clock.time() - start_time))


