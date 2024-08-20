import cv2
import numpy as np
import enum

# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('/Users/adam/Documents/FormFactor/FormFactorTests/content/video/raw-shot/shot-002.mov')


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

    def process_frame(self, input_frame: cv2.Mat, overlay_frame: cv2.Mat):
        height, width = input_frame.shape
        input_frame = input_frame[int(height / 2):height, int(width / 2):width]
        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        line_image = np.copy(overlay_frame)  # creating a blank to draw lines on

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(input_frame, rho, theta, self._sliders["threshold"]["value"], np.array([]),
                                self._sliders["min_line_length"]["value"], self._sliders["max_line_gap"]["value"])

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

        new_box = None

        if len(points) >= 2:
            points.sort()
            max_gap = -1
            selected = (0, 0)
            for i in range(len(points)-1):
                gap = points[i+1] - points[i]
                if max_gap <= gap:
                    max_gap = gap
                    selected = (points[i], points[i+1])

            for s in selected:
                cv2.line(line_image,
                         (0,s),(width,s),
                         (255, 0, 0),
                         2
                         )

            new_box = ((width, selected[0]), (int(.9 * width), selected[1]))
            line_image = cv2.rectangle(line_image, (width, selected[0]), (int(.9 * width), selected[1]), (0, 255, 255), 1)

        cv2.imshow(self.window_name, cv2.resize(line_image, (640, 360)))

        return new_box

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

class BoxStateMachine:

    class State(enum.Enum):
        UNDEFINED = 0
        SET = 1

    BOX_BUFFER_SIZE = 30

    def __init__(self):
        self._state = self.State.UNDEFINED
        self._frame_count = 0
        self._box_buffer = []
        self._current_box = None

    def update(self, new_box):
        # print(f"box is {new_box}")
        self._box_buffer.append(new_box)
        if len(self._box_buffer) > self.BOX_BUFFER_SIZE:
            self._box_buffer.pop(0)

        if self._state == self.State.UNDEFINED:
            if len(self._box_buffer) < 30:
                return None
            # We need to have seen boxes on the last 30 frames to call it "good"
            x0_total = 0
            x1_total = 0
            y0_total = 0
            y1_total = 0
            for new_box in self._box_buffer:
                if not new_box:
                    return None
                x0_total += new_box[0][0]
                x1_total += new_box[1][0]
                y0_total += new_box[0][1]
                y1_total += new_box[1][1]
            self._current_box = (
                (int(x0_total/self.BOX_BUFFER_SIZE), int(y0_total/self.BOX_BUFFER_SIZE)),
                (int(x1_total/self.BOX_BUFFER_SIZE), int(y1_total/self.BOX_BUFFER_SIZE)),
            )
            print(f"Current box is {self._current_box}")
            self._state = self.State.SET
            return self._current_box
        elif self._state == self.State.SET:
            # If the last 30 frames in a row don't have 25 boxes, go back to init stats
            valid_boxes = 0
            x0_total = 0
            x1_total = 0
            y0_total = 0
            y1_total = 0
            for b in self._box_buffer:
                valid_boxes += 1
                x0_total += new_box[0][0]
                x1_total += new_box[1][0]
                y0_total += new_box[0][1]
                y1_total += new_box[1][1]
            if valid_boxes < 25:
                self._current_box = None
                print("Going back to undefined state")
                self._state = self.State.UNDEFINED
                return None
            x0_total += self._current_box[0][0] * valid_boxes
            x1_total += self._current_box[1][0] * valid_boxes
            y0_total += self._current_box[0][1] * valid_boxes
            y1_total += self._current_box[1][1] * valid_boxes

            self._current_box = (
                (int(x0_total/(valid_boxes * 2)), int(y0_total/(valid_boxes * 2))),
                (int(x1_total/(valid_boxes * 2)), int(y1_total/(valid_boxes * 2))),
            )
            return self._current_box

# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening video file")

cv2.namedWindow("raw")
cv2.moveWindow("raw", 0,0)

cv2.namedWindow("bw")
cv2.moveWindow("bw", 645, 0)

cv2.namedWindow("blur")
cv2.moveWindow("blur", 0, 365)

lines_window = LinesWindow()

box_state_machine = BoxStateMachine()

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 360))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kernel_size = 3
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    cv2.imshow("raw", edges)

    box = lines_window.process_frame(edges, frame)

    box = box_state_machine.update(box)

    if box is not None:
        frame = cv2.rectangle(frame, box[0], box[1], (0, 255, 255), 1)

    cv2.imshow("raw", frame)

    cv2.imshow("blur", blur_gray)

    key = cv2.waitKey(1) & 0xFF
    # Press Q on keyboard to exit
    if key == ord('q'):
        break
