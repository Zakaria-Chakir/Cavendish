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
# found_circles = find_contour_mnk(video_input_path = 'Input_videos\Third_video_data_v2.mp4', video_output_path = "Output_videos\Third_test_tracking.avi", 
#                  pos_circle_txt_path = "Position of circles\Third_test_position_of_dot.txt", Countour_finding_param = {
#                             'dp': 1,                   # Inverse of precision (lower = higher precision)
#                             'mindist': 6,             # Minimum distance between circle centers (px)
#                             'canny_threshold': 6,      # Higher = fewer edges (but more precise)
#                             'circle_threshold': 6,     # Higher = stricter circle detection
#                             'min_radius': 1,            # Minimum radius (px)
#                             'max_radius': 5,            # Maximum radius (px)
#                             }, scale_factor = 0.25, lower_color_bdd = np.array([30, 100, 30]), 
#                             upper_color_bdd = np.array([180, 255, 200]), morb_kernel_size = (4,4), real_time_video = True,
#                             mouse_curser = False)



########### Importing data from a textfile ##############
with open("Position of circles\Third_test_position_of_dot.txt", 'r') as file:
    found_circles = [ast.literal_eval(line.strip()) for line in file]
found_circles = np.asarray(found_circles)


######## Visualising data and finding the period ##########
time = [x for x in range(len(found_circles)) if x < len(found_circles)*0.995]
planar_disp = [y for x,y in zip(range(len(found_circles)), found_circles[:,0]) if x < len(found_circles)*0.995]
displacement_err = [y for x,y in zip(range(len(found_circles)), found_circles[:,2]) if x < len(found_circles)*0.995]

# plt.plot(time, planar_disp)
# plt.show()

fft_signal = np.abs(scipy.fft.rfft(planar_disp))
fft_freq = scipy.fft.rfftfreq(len(time), d=1/60)


idx_max, _ = scipy.signal.find_peaks(fft_signal, threshold = 10000, distance = 60, prominence = 100)
print(fft_freq[idx_max])
plt.plot(fft_freq, fft_signal, color = 'gray')
plt.scatter(fft_freq[idx_max], fft_signal[idx_max], color = 'orange', marker = 'x')
plt.show()



##### Curve fit #####

coeff, coeff_err = curve_fitting(time, planar_disp, y_err = displacement_err, x_title = 'Time (s)', y_title = 'Planar displacement (px)', model = torsional_oscillator_model)
for c,e in zip(coeff, coeff_err) :
    print(c, '+/-', e)


print()
print("--- %s seconds ---" % (clock.time() - start_time))


