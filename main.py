# importing libraries
import cv2

# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('/Users/adam/Documents/bowling/video/20240720-1/1.MOV')

# Check if camera opened successfully
if cap.isOpened() is False:
    print("Error opening video file")

    # Read until video is completed
x = 0
paused = False
one_frame = False
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
start_time_ms = 0
while cap.isOpened():

    x += 1

    # Capture frame-by-frame

    if not paused or one_frame:
        one_frame = False
        ret, frame = cap.read()
        if ret == True:

            cv2.putText(frame, text = f"Start time: {str(start_time_ms)}",
                org = (200, 200),
                fontFace = cv2.FONT_HERSHEY_DUPLEX,
                fontScale = 2.0,
                color = (125, 246, 55),
                thickness = 3)
            current_time_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
            cv2.putText(frame, text = f"Current time: {current_time_ms}",
                org = (200, 300),
                fontFace = cv2.FONT_HERSHEY_DUPLEX,
                fontScale = 2.0,
                color = (125, 246, 55),
                thickness = 3)
            delta_ms = (current_time_ms - start_time_ms) / 1000.0
            if delta_ms == 0:
                rpm = 0
            else:
                rpm = 60 / delta_ms
            cv2.putText(frame, text = f"RPM: {rpm}",
                org = (200, 400),
                fontFace = cv2.FONT_HERSHEY_DUPLEX,
                fontScale = 2.0,
                color = (125, 246, 55),
                thickness = 3)
            # Display the resulting frame
            cv2.imshow('Frame', frame)
        # Break the loop
        else:
            break

    key = cv2.waitKey(25) & 0xFF
    # Press Q on keyboard to exit
    if key == ord('q'):
        break
    elif key == ord(' '):
        # pause when space key is pressed
        paused = not paused
    elif key == ord(']'):
        # advance one frame
        one_frame = True
        paused = True
    elif key == ord('='):
        # advance 5 frames
        one_frame = True
        paused = True
        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number + 4)
    elif key == ord('['):
        # reverse one frame
        one_frame = True
        paused = True
        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 2)
    elif key == ord('-'):
        # reverse 5 frames
        one_frame = True
        paused = True
        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 6)
    elif key == ord('s'):
        # mark start frame and rerender this frame to update the text
        start_time_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
        one_frame = True
        paused = True
        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)





# When everything done, release
# the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
