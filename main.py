import cv2 as cv
import numpy as np
import scipy
from dot_detection import find_contour_mnk
import matplotlib.pyplot as plt
import csv
import ast
import webbrowser
import time
import PIL as pillow

start_time = time.time()

########### Video tracking function ###############
found_circles = find_contour_mnk(video_input_path = 'Input_videos\Cropped_first_data_vid.mp4', video_output_path = "Output_videos\First_test_tracking.avi", 
                 pos_circle_txt_path = "Position of circles\Calibration_test_position_of_dot.txt", Countour_finding_param = {
                            'dp': 1,                   # Inverse of precision (lower = higher precision)
                            'mindist': 11,              # Minimum distance between circle centers (px)
                            'canny_threshold': 1,      # Higher = fewer edges (but more precise)
                            'circle_threshold': 1,     # Higher = stricter circle detection
                            'min_radius': 1,            # Minimum radius (px)
                            'max_radius': 5,            # Maximum radius (px)
                            }, scale_factor = 0.25, lower_color_bdd = np.array([50, 120, 50]), 
                            upper_color_bdd = np.array([180, 255, 200]), morb_kernel_size = False, real_time_video = True,
                            mouse_curser = True)



########### Importing data from a textfile ##############
with open("Position of circles\Calibration_test_position_of_dot.txt", 'r') as file:
    found_circles = [ast.literal_eval(line.strip()) for line in file]
found_circles = np.asarray(found_circles)


######## Visualising data and finding the period ##########
plt.plot([x/60 for x in range(len(found_circles))], found_circles[:,0])
plt.show()

fft_signal = scipy.fft.rfft(found_circles[:,0])
fft_freq = scipy.fft.rfftfreq(len(found_circles), d=1/60)


idx_max, _ = scipy.signal.find_peaks(fft_signal, threshold = 10000, distance = 60, prominence = 100)
# print(fft_freq[idx_max])
plt.plot(fft_freq, fft_signal, color = 'gray')
plt.scatter(fft_freq[idx_max], fft_signal[idx_max], color = 'orange', marker = 'x')
plt.show()


print("--- %s seconds ---" % (time.time() - start_time))


