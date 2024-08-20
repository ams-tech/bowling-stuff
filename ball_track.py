import cv2
import numpy as np

# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('/Users/adam/bowl3.mov')


# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening video file")


while cap.isOpened():
    ret, frame = cap.read()

    # Convert to grayscale.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur using 3 * 3 kernel.
    gray_blurred = cv2.blur(gray, (3, 3))



    minDist = 100
    param1 = 500  # 500
    param2 = 40  # 200 #smaller value-> more false circles
    minRadius = 2
    maxRadius = 150  # 10

    # docstring of HoughCircles: HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles
    detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)

    # Draw circles that are detected.
    if detected_circles is not None:

        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))

        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]

            # Draw the circumference of the circle.
            cv2.circle(frame, (a, b), r, (0, 255, 0), 2)

            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(frame, (a, b), 1, (0, 0, 255), 3)

    # show the base frame
    cv2.imshow('frame', frame)

    key = cv2.waitKey(1) & 0xFF
    # Press Q on keyboard to exit
    if key == ord('q'):
        break
