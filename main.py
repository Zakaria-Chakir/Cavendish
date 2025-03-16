import cv2 as cv
import numpy as np
from dot_detection import find_contour_mnk

find_contour_mnk(video_input_path = 'Input_videos\Cropped_first_data_vid.mp4', video_output_path = "Output_videos\First_test_tracking.avi", 
                 pos_circle_txt_path = "Position of circles\Calibration_test_position_of_dot.txt", Countour_finding_param = {
                            'dp': 1,                   # Inverse of precision (lower = higher precision)
                            'mindist': 11,              # Minimum distance between circle centers (px)
                            'canny_threshold': 1,      # Higher = fewer edges (but more precise)
                            'circle_threshold': 1,     # Higher = stricter circle detection
                            'min_radius': 1,            # Minimum radius (px)
                            'max_radius': 5,            # Maximum radius (px)
                            }, scale_factor = 0.25, lower_color_bdd = np.array([50, 120, 50]), 
                            upper_color_bdd = np.array([180, 255, 200]), morb_kernel_size = False, real_time_video = True,
                            mouse_curser = False)

