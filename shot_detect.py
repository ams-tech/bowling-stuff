import cv2
import numpy as np

# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('/Users/adam/Documents/FormFactor/FormFactorTests/content/video/raw-shot/shot-001.mov')


class CvWindow:
    def __init__(self, window_name: str, window_location_x: int = 0, window_location_y: int = 0, sliders = None):
        self.window_name = window_name
        self.window_name = window_name
        self.sliders = sliders
        if sliders is None:
            self.sliders = {}
        cv2.namedWindow(window_name)
        cv2.moveWindow(window_name, window_location_x, window_location_y)
        for s in self.sliders.values():
            def _callback(val):
                s["value"] = val
            cv2.createTrackbar(s["title"], window_name, s["value"], s["max"], _callback)


class LinesWindow(CvWindow):
    def __init__(self, window_name="Lines"):
        self._sliders = {
            "min_line_length": {"value": 50, "max": 300, "title": "Min Line Length"},
            "max_line_gap": {"value": 30, "max": 300, "title": "Max Line Gap"},
            "threshold": {"value": 20, "max": 300, "title": "Hough Threshold"},
        }
        super().__init__(window_name, sliders=self._sliders)

    def _process_frame(self, input_frame: cv2.Mat, overlay_frame: cv2.Mat) -> cv2.Mat:
        height, width = input_frame.shape
        input_frame = input_frame[int(height / 2):height, int(width / 2):width]
        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        line_image = np.copy(overlay_frame)  # creating a blank to draw lines on

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(input_frame, rho, theta, self._sliders["threshold"]["value"], np.array([]),
                                self._sliders["min_line_length"]["value"], self._sliders["max_line_gap"]["value"])

        x = 0
        if lines is None:
            lines = []

        points = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.2:
                    if x1 >= .95 * (width / 2):
                        points.append(int(height / 2) + y1)
                    if x2 >= .95 * (width / 2):
                        points.append(int(height / 2) + y2)

        for p in points:
            cv2.line(line_image,
                     (0,p),(width,p),
                     (255, 0, 0),
                     5
                     )
        cv2.imshow(self.window_name, cv2.resize(line_image, (640, 360)))

class HoughCirclesWindow(CvWindow):
    def __init__(self, window_name="Circles"):
        self._sliders = {
            "param1": {"value": 14, "max": 100, "title": "Parameter 1"},
            "param2": {"value": 25, "max": 100, "title": "Parameter 2"},
            "minRadius": {"value": 10, "max": 50, "title": "Minimum Radius"},
            "maxRadius": {"value": 40, "max": 50, "title": "Maximum Radius"},
        }
        super().__init__(window_name, sliders=self._sliders)

    def _process_frame(self, input_frame: cv2.Mat, overlay_frame: cv2.Mat) -> cv2.Mat:
        new_frame = overlay_frame.copy()

        detected_circles = cv2.HoughCircles(
            input_frame,
            cv2.HOUGH_GRADIENT,
            1,
            10,
            param1=self._sliders["param1"]["value"],
            param2=self._sliders["param2"]["value"],
            minRadius=self._sliders["minRadius"]["value"],
            maxRadius=self._sliders["maxRadius"]["value"]
        )

        # Draw circles that are detected.
        if detected_circles is not None:

            # Convert the circle parameters a, b and r to integers.
            detected_circles = np.uint16(np.around(detected_circles))

            for pt in detected_circles[0, :]:
                a, b, r = pt[0], pt[1], pt[2]

                # Draw the circumference of the circle.
                cv2.circle(new_frame, (a, b), r, (0, 255, 0), 2)

                # Draw a small circle (of radius 1) to show the center.
                cv2.circle(new_frame, (a, b), 1, (0, 0, 255), 3)

        cv2.imshow(self.window_name, cv2.resize(new_frame, (640, 360)))

        return new_frame


# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening video file")

cv2.namedWindow("raw")
cv2.moveWindow("raw", 0,0)

cv2.namedWindow("bw")
cv2.moveWindow("bw", 645, 0)

cv2.namedWindow("blur")
cv2.moveWindow("blur", 0, 365)

circles_window = LinesWindow()

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 360))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kernel_size = 13
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    low_threshold = 50
    high_threshold = 100
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)


    cv2.imshow("raw", edges)

    circles_window._process_frame(edges, frame)

    cv2.imshow("blur", blur_gray)

    key = cv2.waitKey(1) & 0xFF
    # Press Q on keyboard to exit
    if key == ord('q'):
        break
