
############################################################

# Note : this code is by no means optimal, but it works for now, I'll optimize it as we take more and more data and 
# as I see how much optimization it actually needs.
import cv2 as cv
import numpy as np

# Statistics of number of found circles
num_of_extra_circles = 0 # this counts the number of frame were we found more that on point (very bad if bigger than 0)
num_of_not_found_circle = 0 # this counts the frames were we found nothing 
# (not that bad as long as it is low, let's say less than 10% of num of frames total)


# Function to print the position of the mouse if it is moved 
def mouse_callback(event, x, y, i, j): # idk what the 2 last param do and idc, i dont need them)
    if event == cv.EVENT_MOUSEMOVE:
        print(f"Mouse position: ({x}, {y})")

# Read the input video file
cap = cv.VideoCapture('20250312_143444 (2).mp4')
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

# Scaling down the output (better to see it like that)
scale_factor = 0.25

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'XVID')  # Use XVID codec for better compatibility
out = cv.VideoWriter('Tracking_dot.avi', fourcc, fps, (int(width * scale_factor), int(height * scale_factor)))

# # Get the position of stuff on the window
# cv.namedWindow("Frame")
# cv.setMouseCallback("Frame", mouse_callback)



# Keeping track of the circles I found, format of element of the array [x,y,r] center position and then radius
found_circles = []

while cap.isOpened():
    isread, frame = cap.read() # isread litteraly says if the video way read correctly, frame is the frame

    # Error handling (case were frame cant be read)
    frame_count += 1
    if not isread:
        if frame_count >= total_frames - 1:
            print("Stream end! Exiting ...")
            break
        else : 
            print('A corrupt frame was given. Exiting...')
            break

    # Resize the given frame
    frame = cv.resize(frame, (int(width * scale_factor), int(height * scale_factor)))

    # Convert to HSV color space (apparently this is ideally for color tracking)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Bounds for our laser
    lower_bound = np.array([0, 200, 0])  # Lower bound for brightness (close to white)
    upper_bound = np.array([255, 255, 255])  # Upper bound for brightness (white)

    # Create a mask based on the threshold (converts the map to binary and takes off any colors not in the range)
    mask = cv.inRange(hsv, lower_bound, upper_bound)

    # If you want to use this (to like blur unexpected detected pts, we need to use find contours or Canny)
    '''  # Addition if not perfect convergence (but could also lead to have nothing detected)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20)) # use a large kernel since dot is small
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)'''

    # Find circles using Hough Circle Transform (Contour getting method similar to Canny but for circles)
    # keep param1 and param2 low since the mask already does a very good job. 
    # The other parameters were obtained by trail and error
    circles = cv.HoughCircles(mask, cv.HOUGH_GRADIENT, dp=1, minDist=40, param1=10, param2=10, minRadius=5, maxRadius=60)
    ''' this function outputs an array of this format : [[[x1, y1,r1], [x2, y2, r2], ...]] Apparently they use double brackets 
    to keep being consistent with other functions'''

    if circles is not None:
        if len(circles[0]) == 1 :
            # this takes off the extra brackets
            circle = np.uint16(np.around(circles[0, :]))  # round coordinates of found circle to integers to represent the pixel position
            # Draw the circles on the frame
            x, y, r = circle 
            cv.circle(frame, (x, y), r, (0, 0, 255), 4)  # Draw circle
            cv.rectangle(frame, (x - 1, y - 1), (x + 1, y + 1), (255, 128, 0), -1)  # Draw center 

            found_circles.append([x,y,r])



        elif len(circle[0]) > 1 : 
            print(f"{len(circle)} circles detected, Warning")
            num_of_extra_circles += 1
        
        else :
            print('IDK what happened this is not suppose to be passed here')
    else:
        print("No circles detected.")
        num_of_not_found_circle += 1


    # Write the frame to output video
    out.write(frame)

    # Display the frame for immediate feedback
    cv.imshow("Tracking_real_time_video", frame)

    # # To analyze frame by frame the video
    # # Wait for keypress to continue to the next frame
    # key = cv.waitKey(0)  # This waits indefinitely for a key press

    # if key == ord('f'):  # Proceed to the next frame when 'f' is pressed
    #     continue

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv.destroyAllWindows()

print(f'Total number of frames : {total_frames}')
print(f'The number of frames were more than 1 circle were found is : {num_of_extra_circles} or {round(100*num_of_extra_circles/total_frames, 3)}%')
print(f'The number of frames with no circles found is : {num_of_not_found_circle} or {round((100*num_of_not_found_circle)/total_frames, 3)}%')


