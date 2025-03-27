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
# total_time, found_circles = find_contour_mnk(video_input_path = r'Input_videos\20250325_125037.mp4', video_output_path = r"Output_videos\fourth_video_data.avi", 
#                  pos_circle_txt_path = r"Position of circles\fourth_video_data.txt", Countour_finding_param = {
#                             'dp': 1,                   # Inverse of precision (lower = higher precision)
#                             'mindist': 10,             # Minimum distance between circle centers (px)
#                             'canny_threshold': 6,      # Higher = fewer edges (but more precise)
#                             'circle_threshold':6,     # Higher = stricter circle detection
#                             'min_radius': 1,            # Minimum radius (px)
#                             'max_radius': 7,            # Maximum radius (px)
#                             }, scale_factor = 0.25, lower_color_bdd = np.array([60, 100, 60]), 
#                             upper_color_bdd = np.array([240, 255, 250]), morb_kernel_size = (4,4), real_time_video = True,
#                             mouse_curser = True, crop_xmin = 330, crop_xmax = 460, crop_ymin = 500, crop_ymax = 1360)



########### Importing data from a textfile ##############
with open(r"Position of circles\fourth_video_data.txt", 'r') as file:
    found_circles = [ast.literal_eval(line.strip()) for line in file]
    total_time = [i/30 for i in range(155395)]
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


#### Fourier transform of the signal to find the period ######
fft_signal = np.abs(scipy.fft.rfft(planar_disp))
fft_freq = scipy.fft.rfftfreq(len(time), d=1/60) 


idx_max, _ = scipy.signal.find_peaks(fft_signal, threshold = 10000, distance = 60, prominence = 100) # this finds the peaks in fourier space
print(fft_freq[idx_max])
plt.plot(fft_freq, fft_signal, color = 'gray')
plt.scatter(fft_freq[idx_max], fft_signal[idx_max], color = 'orange', marker = 'x')
plt.title(f'Fourier transform')
plt.show()



##### Curve fit #####
Bounds = [[10, 10**-8, 0, 0, -np.pi, 0, 8*10**-13, -1000], [2000, 10**-2, 10**-2, 10**-5,np.pi, 2000, 2*10**-5, 1000]]
Initial_guess = [100, 10**-3,0.00000000001, 7*10**-11, 0, np.mean(planar_disp[10:]), 8*10**-8, 900]
coeff, coeff_err = curve_fitting(time[10:], planar_disp[10:] , y_err = displacement_err[10:], x_title = 'Time (s)', 
                                 y_title = 'Angular displacement (rad)', model = torsional_oscillator_model,bdds = Bounds, 
                                 initial_guess = Initial_guess) # see function in oscillation model


# Print the parameters
for c,e, name, maximum, minimum in zip(coeff, coeff_err, ['C', 'I' ,'damp', 'G', 'phi', 'k', 'kappa','h'], 
            Bounds[0], Bounds[1]) :
    print(f"{name} : {c:.4e} +/- {e:.4e}. Max : {maximum:.4e}     . Min : {minimum:.4e}  ")


print()
print("--- %s seconds ---" % (clock.time() - start_time))


