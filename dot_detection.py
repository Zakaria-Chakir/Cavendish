# Note : this code is by no means optimal, but it works for now, I'll optimize it as we take more and more data and 
# as I see how much optimization it actually needs.
import cv2 as cv
import numpy as np
import time as clock

def find_contour_mnk(video_input_path : str, video_output_path :str, pos_circle_txt_path : str, Countour_finding_param = {
                            'dp': 1,                   # Inverse of precision (lower = higher precision)
                            'mindist': 40,              # Minimum distance between circle centers (px)
                            'canny_threshold': 10,      # Higher = fewer edges (but more precise)
                            'circle_threshold': 10,     # Higher = stricter circle detection
                            'min_radius': 5,            # Minimum radius (px)
                            'max_radius': 60            # Maximum radius (px)
                            }, scale_factor = 0.25, lower_color_bdd = np.array([0, 200, 0]), 
                            upper_color_bdd = np.array([255, 255, 255]), morb_kernel_size = (10, 10), real_time_video = True,
                            mouse_curser = False, crop_xmin = 0, crop_xmax = -1, crop_ymin = 0, crop_ymax = -1):
    
    """
    Process a video to detect and track circular objects using color filtering and Hough Circle Transform.
    
    The function processes each video frame to:
    1. Resize frames according to scale factor
    2. Convert to HSV color space and apply color thresholding
    3. Perform morphological operations to clean up the mask
    4. Detect circles using Hough Circle Transform
    5. Draw detected circles and save results
    6. Output statistics about detection accuracy

    Parameters:
    -----------
    video_input_path : str
        Path to input video file
    video_output_path : str
        Path for output video with detected circles visualized
    pos_circle_txt_path : str
        Path to save detected circle positions (format: [x, y, radius] per line)
    Countour_finding_param : dict, optional
        Dictionary of parameters for circle detection:
        - dp: Inverse ratio of accumulator resolution (lower = higher precision)
        - mindist: Minimum distance between circle centers in pixels
        - canny_threshold: Higher values reduce detected edges but increase precision
        - circle_threshold: Higher values require clearer circle candidates
        - min_radius: Minimum circle radius to detect (pixels)
        - max_radius: Maximum circle radius to detect (pixels)
    scale_factor : float, optional
        Scaling factor for resizing frames (0-1)
    lower_color_bdd : np.ndarray, optional
        Lower bound for HSV color thresholding (BGR format)
    upper_color_bdd : np.ndarray, optional
        Upper bound for HSV color thresholding (BGR format)
    morb_kernel_size : tuple, optional
        Kernel size for morphological closing operation (width, height)
    real_time_video : bool, optional
        Enable real-time video display during processing
    mouse_curser : bool, optional
        Enable mouse position tracking in display window

    Returns:
    --------
    list
        List of detected circles in format [[x1, y1, r1], [x2, y2, r2], ...]

    Notes:
    ------
    - Output video uses XVID codec for compatibility
    - Statistics about detection accuracy are printed to console
    - Processing can be interrupted with 'q' key during real-time display
    - Requires OpenCV and NumPy dependencies

    Example:
    --------
    >>> params = {
        'dp': 1,
        'mindist': 40,
        'canny_threshold': 10,
        'circle_threshold': 10,
        'min_radius': 5,
        'max_radius': 60
    }
    >>> circles = find_contour_mnk(
        'input.mp4',
        'output.avi',
        'positions.txt',
        Countour_finding_param=params,
        scale_factor=0.5
    )
    """

    # Statistics of number of found circles
    num_of_extra_circles = 0 # this counts the number of frame were we found more that on point (very bad if bigger than 0)
    num_of_not_found_circle = 0 # this counts the frames were we found nothing 
    # (not that bad as long as it is low, let's say less than 10% of num of frames total)


    # Function to print the position of the mouse if it is moved 
    def mouse_callback(event, x, y, i, j): # idk what the 2 last param do and idc, i dont need them)
        if event == cv.EVENT_LBUTTONDOWN:
            print(f"Mouse position: ({x}, {y})")

    # Read the input video file
    cap = cv.VideoCapture(video_input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()


    # Total number of frame (used for statistics)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    frame_count = 0 # I count the frames for error handling

    # Get basic info on the input video
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)
    print(f'{fps}fps ')
    print(total_frames, 'frames')

    # # Define the codec and create VideoWriter object
    # fourcc = cv.VideoWriter_fourcc(*'XVID')  # Use XVID codec for better compatibility
    # out = cv.VideoWriter(video_output_path, fourcc, int(np.ceil(fps+1)), (int(crop_xmax-crop_xmin), int(crop_ymax-crop_ymin)))

    # # Get the position of stuff on the window
    if mouse_curser :
        cv.namedWindow("Tracking_real_time_video")
        cv.setMouseCallback("Tracking_real_time_video", mouse_callback)



    # Keeping track of the circles I found, format of element of the array [x,y,r] center position and then radius
    found_circles = []
    start_time = clock.time()
    while cap.isOpened():
        isread, frame = cap.read() # isread litteraly says if the video way read correctly, frame is the frame
        if frame_count == 0 : print(frame.shape)
        # Error handling (case were frame cant be read)
        frame_count += 1
        if not isread:
            if frame_count >= total_frames - 1:
                print("Stream end! Exiting ...")
                break
            else : 
                print('A corrupt frame was given. Exiting...')
                break
        
        if frame_count % (1000) == 0 : 
            print("--- %s seconds ---" % (clock.time() - start_time))
            print('frame number :', frame_count, 'out of', total_frames)
            start_time = clock.time()


        # Resize the given frame
        # frame = frame[crop_xmin:crop_xmax, crop_ymin: crop_ymax]
        
        

        # Convert to HSV color space (apparently this is ideally for color tracking)
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # Create a mask based on the threshold (converts the map to binary and takes off any colors not in the range)
        mask = cv.inRange(hsv, lower_color_bdd, upper_color_bdd)

        # If you want to use this (to like blur unexpected detected pts, we need to use find contours or Canny)
        # Addition if not perfect convergence (but could also lead to have nothing detected)
        if morb_kernel_size :
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, morb_kernel_size) # use a large kernel since dot is small
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

        # Find circles using Hough Circle Transform (Contour getting method similar to Canny but for circles)
        # keep param1 and param2 low since the mask already does a very good job. 
        # The other parameters were obtained by trail and error

        # Extracting the value of the parameters from the input of the function
        dp = Countour_finding_param['dp'] # inverse of precision
        min_dist = Countour_finding_param['mindist'] # min dist between circles (in px)
        threshold_canny = Countour_finding_param['canny_threshold'] # high means less edges are detected but also more precise
        threshold_circle  = Countour_finding_param['circle_threshold'] # high means only very clear circles are detected
        min_r = Countour_finding_param['min_radius'] # minimal radius of the circles (in px)
        max_r = Countour_finding_param['max_radius'] # maximal radius of the circles (in px)
        circles = cv.HoughCircles(mask, cv.HOUGH_GRADIENT, dp=dp, minDist=min_dist, param1=threshold_canny, 
                                  param2=threshold_circle, minRadius=min_r, maxRadius=max_r)
        # this function outputs an array of this format : [[[x1, y1,r1], [x2, y2, r2], ...]] Apparently they use double brackets 
        # to keep being consistent with other functions

        if circles is not None:
            if len(circles[0]) == 1 :
                # this takes off the extra brackets
                circle = np.uint16(np.around(circles[0, :]))  # round coordinates of found circle to integers to represent the pixel position
                # Draw the circles on the frame
                x, y, r = circle[0] 
                cv.circle(frame, (x, y), r, (0, 0, 255), 4)  # Draw circle
                cv.rectangle(frame, (x - 1, y - 1), (x + 1, y + 1), (255, 128, 0), -1)  # Draw center 

                found_circles.append([x,y,r])



            elif len(circles[0]) > 1 : 
                circle = np.uint16(np.around(circles[0, :]))
                # print(f"{len(circles[0])} circles detected, Warning")
                num_of_extra_circles += 1
                for x,y,r in circle:
                    cv.circle(frame, (x, y), r, (255, 0, 0), 4)  # Draw circle
                    cv.rectangle(frame, (x - 1, y - 1), (x + 1, y + 1), (255, 128, 0), -1)  # Draw center 
                
                found_circles.append([np.mean(circle[:,0]),np.mean(circle[:,1]),np.mean(circle[:,2])])
            
            else :
                print('IDK what happened this is not suppose to be passed here')
        else:
            # print("No circles detected.")
            num_of_not_found_circle += 1
            found_circles.append([None, None, None])


        # Write the frame to output video
        # out.write(frame)

        if real_time_video :
            # Display the frame for immediate feedback
            # frame = cv.resize(frame, (int(width * scale_factor), int(height * scale_factor)))
            cv.imshow("Tracking_real_time_video", frame)
            cv.imshow('Color mask', mask)

            if cv.waitKey(1) == ord('q'):
                break

    cap.release()
    # out.release()
    cv.destroyAllWindows()

    print(f'Total number of frames : {total_frames}')
    print(f'The number of frames were more than 1 circle were found is : {num_of_extra_circles} or {round(100*num_of_extra_circles/total_frames, 3)}%')
    print(f'The number of frames with no circles found is : {num_of_not_found_circle} or {round((100*num_of_not_found_circle)/total_frames, 3)}%')
    print()

    with open(pos_circle_txt_path, 'w') as f:
        for entry in found_circles :
            f.write(str(entry)+'\n')

    time = [i/round(fps) for i in range(total_frames)]

    return found_circles, time