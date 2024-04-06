import argparse
import copy
import cv2 as cv
from utils import CvFpsCalc


# Handles camera initialization and operation
class Camera:
    def __init__(self):
        self.argument_parsing()
        self.initialize_camera()
        self.cvFpsCalc = CvFpsCalc(buffer_len=10)
        self.mode = 0

    # Get args from command
    def get_args(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--device", type=int, default=0)
        parser.add_argument("--width", help="cap width", type=int, default=960)
        parser.add_argument("--height", help="cap height", type=int, default=540)

        parser.add_argument("--use_static_image_mode", action="store_true")
        parser.add_argument(
            "--min_detection_confidence",
            help="min_detection_confidence",
            type=float,
            default=0.7,
        )
        parser.add_argument(
            "--min_tracking_confidence",
            help="min_tracking_confidence",
            type=int,
            default=0.5,
        )

        args = parser.parse_args()

        return args

    # Parse command arguments
    def argument_parsing(self):
        args = self.get_args()  # Retrieve command line arguments

        # Set class attributes
        self.cap_device = args.device
        self.cap_width = args.width
        self.cap_height = args.height
        self.use_static_image_mode = args.use_static_image_mode
        self.min_detection_confidence = args.min_detection_confidence
        self.min_tracking_confidence = args.min_tracking_confidence
        self.use_brect = True

    # Initialize camera object
    def initialize_camera(self):
        #remove cv.CAP_DSHOW if camera is not opening
        self.cap = cv.VideoCapture(self.cap_device,cv.CAP_DSHOW)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.cap_width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.cap_height)

    ## CAMERA CAPTURE #############
    # Initiates capturing and pre-processing of frame
    def capture(self):
        self.image, self.debug_image = self.capture_and_preprocess_frame()
        # Detection implementation
        self.image = cv.cvtColor(self.image, cv.COLOR_BGR2RGB)
        self.image.flags.writeable = False  # set to read only

    # Captures and pre-processes frame
    def capture_and_preprocess_frame(self):
        ret, image = self.cap.read()
        if not ret:
            return None, None
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)
        return image, debug_image

    ## FPS CALCULATIONS ############
    # Calculate FPS using last 10 buffers
    def calculate_fps(self):
        self.fps = self.cvFpsCalc.get()

    ## KEYBOARD KEY CHECKER ############
    # Handles flow for if ESC or other key was pressed
    def key_check(self):
        self.process_key(cv)
        if self.key is None:  # ESC pressed
            return False  # Signal to stop the loop
        self.select_mode()
        return True  # Continue loop

    # Actually checks what key was pressed
    def process_key(self, cv):
        self.key = cv.waitKey(10) & 0xFF  # Use mask for compatibility
        if self.key == 27:  # ESC
            self.key = None

    # Selects a mode depending on what key is pressed
    def select_mode(self):
        self.number = -1
        if 48 <= self.key <= 57:  # 0 ~ 9
            self.number = self.key - 48
        elif self.key == 113:  # using q for 10
            self.number = 10
        elif self.key == 119:  # using w for 11
            self.number = 11
        elif self.key == 101:  # using e for 12  UNUSED CURRENTLY
            self.number = 12
        elif self.key == 114:  # using r for 13  UNUSED CURRENTLY
            self.number = 13
        if self.key == 110:  # n
            self.mode = 0
        if self.key == 107:  # k
            self.mode = 1
        if self.key == 104:  # h
            self.mode = 2

    ## DRAWING CAMERA STUFF ###############
    def draw_bounding_rect(self, brect):
        if self.use_brect:
            # Outer rectangle
            cv.rectangle(
                self.debug_image,
                (brect[0], brect[1]),
                (brect[2], brect[3]),
                (0, 0, 0),
                1,
            )

    def draw_info_text(
        self, image, brect, handedness, hand_sign_text, finger_gesture_text
    ):
        cv.rectangle(
            image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1
        )

        info_text = handedness.classification[0].label[0:]
        if hand_sign_text != "":
            info_text = info_text + ":" + hand_sign_text
        cv.putText(
            image,
            info_text,
            (brect[0] + 5, brect[1] - 4),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv.LINE_AA,
        )

        if finger_gesture_text != "":
            cv.putText(
                image,
                "Finger Gesture:" + finger_gesture_text,
                (10, 60),
                cv.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 0),
                4,
                cv.LINE_AA,
            )
            cv.putText(
                image,
                "Finger Gesture:" + finger_gesture_text,
                (10, 60),
                cv.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv.LINE_AA,
            )

        return image

    def draw_landmarks(self, image, landmark_point):
        if len(landmark_point) > 0:
            # Thumb
            cv.line(
                image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 0, 0), 6
            )
            cv.line(
                image,
                tuple(landmark_point[2]),
                tuple(landmark_point[3]),
                (255, 255, 255),
                2,
            )
            cv.line(
                image, tuple(landmark_point[3]), tuple(landmark_point[4]), (0, 0, 0), 6
            )
            cv.line(
                image,
                tuple(landmark_point[3]),
                tuple(landmark_point[4]),
                (255, 255, 255),
                2,
            )

            # Index finger
            cv.line(
                image, tuple(landmark_point[5]), tuple(landmark_point[6]), (0, 0, 0), 6
            )
            cv.line(
                image,
                tuple(landmark_point[5]),
                tuple(landmark_point[6]),
                (255, 255, 255),
                2,
            )
            cv.line(
                image, tuple(landmark_point[6]), tuple(landmark_point[7]), (0, 0, 0), 6
            )
            cv.line(
                image,
                tuple(landmark_point[6]),
                tuple(landmark_point[7]),
                (255, 255, 255),
                2,
            )
            cv.line(
                image, tuple(landmark_point[7]), tuple(landmark_point[8]), (0, 0, 0), 6
            )
            cv.line(
                image,
                tuple(landmark_point[7]),
                tuple(landmark_point[8]),
                (255, 255, 255),
                2,
            )

            # Middle finger
            cv.line(
                image, tuple(landmark_point[9]), tuple(landmark_point[10]), (0, 0, 0), 6
            )
            cv.line(
                image,
                tuple(landmark_point[9]),
                tuple(landmark_point[10]),
                (255, 255, 255),
                2,
            )
            cv.line(
                image,
                tuple(landmark_point[10]),
                tuple(landmark_point[11]),
                (0, 0, 0),
                6,
            )
            cv.line(
                image,
                tuple(landmark_point[10]),
                tuple(landmark_point[11]),
                (255, 255, 255),
                2,
            )
            cv.line(
                image,
                tuple(landmark_point[11]),
                tuple(landmark_point[12]),
                (0, 0, 0),
                6,
            )
            cv.line(
                image,
                tuple(landmark_point[11]),
                tuple(landmark_point[12]),
                (255, 255, 255),
                2,
            )

            # Ring finger
            cv.line(
                image,
                tuple(landmark_point[13]),
                tuple(landmark_point[14]),
                (0, 0, 0),
                6,
            )
            cv.line(
                image,
                tuple(landmark_point[13]),
                tuple(landmark_point[14]),
                (255, 255, 255),
                2,
            )
            cv.line(
                image,
                tuple(landmark_point[14]),
                tuple(landmark_point[15]),
                (0, 0, 0),
                6,
            )
            cv.line(
                image,
                tuple(landmark_point[14]),
                tuple(landmark_point[15]),
                (255, 255, 255),
                2,
            )
            cv.line(
                image,
                tuple(landmark_point[15]),
                tuple(landmark_point[16]),
                (0, 0, 0),
                6,
            )
            cv.line(
                image,
                tuple(landmark_point[15]),
                tuple(landmark_point[16]),
                (255, 255, 255),
                2,
            )

            # Little finger
            cv.line(
                image,
                tuple(landmark_point[17]),
                tuple(landmark_point[18]),
                (0, 0, 0),
                6,
            )
            cv.line(
                image,
                tuple(landmark_point[17]),
                tuple(landmark_point[18]),
                (255, 255, 255),
                2,
            )
            cv.line(
                image,
                tuple(landmark_point[18]),
                tuple(landmark_point[19]),
                (0, 0, 0),
                6,
            )
            cv.line(
                image,
                tuple(landmark_point[18]),
                tuple(landmark_point[19]),
                (255, 255, 255),
                2,
            )
            cv.line(
                image,
                tuple(landmark_point[19]),
                tuple(landmark_point[20]),
                (0, 0, 0),
                6,
            )
            cv.line(
                image,
                tuple(landmark_point[19]),
                tuple(landmark_point[20]),
                (255, 255, 255),
                2,
            )

            # Palm
            cv.line(
                image, tuple(landmark_point[0]), tuple(landmark_point[1]), (0, 0, 0), 6
            )
            cv.line(
                image,
                tuple(landmark_point[0]),
                tuple(landmark_point[1]),
                (255, 255, 255),
                2,
            )
            cv.line(
                image, tuple(landmark_point[1]), tuple(landmark_point[2]), (0, 0, 0), 6
            )
            cv.line(
                image,
                tuple(landmark_point[1]),
                tuple(landmark_point[2]),
                (255, 255, 255),
                2,
            )
            cv.line(
                image, tuple(landmark_point[2]), tuple(landmark_point[5]), (0, 0, 0), 6
            )
            cv.line(
                image,
                tuple(landmark_point[2]),
                tuple(landmark_point[5]),
                (255, 255, 255),
                2,
            )
            cv.line(
                image, tuple(landmark_point[5]), tuple(landmark_point[9]), (0, 0, 0), 6
            )
            cv.line(
                image,
                tuple(landmark_point[5]),
                tuple(landmark_point[9]),
                (255, 255, 255),
                2,
            )
            cv.line(
                image, tuple(landmark_point[9]), tuple(landmark_point[13]), (0, 0, 0), 6
            )
            cv.line(
                image,
                tuple(landmark_point[9]),
                tuple(landmark_point[13]),
                (255, 255, 255),
                2,
            )
            cv.line(
                image,
                tuple(landmark_point[13]),
                tuple(landmark_point[17]),
                (0, 0, 0),
                6,
            )
            cv.line(
                image,
                tuple(landmark_point[13]),
                tuple(landmark_point[17]),
                (255, 255, 255),
                2,
            )
            cv.line(
                image, tuple(landmark_point[17]), tuple(landmark_point[0]), (0, 0, 0), 6
            )
            cv.line(
                image,
                tuple(landmark_point[17]),
                tuple(landmark_point[0]),
                (255, 255, 255),
                2,
            )

        # Key Points
        for index, landmark in enumerate(landmark_point):
            if index == 0:  # 手首1
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 1:  # 手首2
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 2:  # 親指：付け根
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 3:  # 親指：第1関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 4:  # 親指：指先
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 5:  # 人差指：付け根
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 6:  # 人差指：第2関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 7:  # 人差指：第1関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 8:  # 人差指：指先
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 9:  # 中指：付け根
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 10:  # 中指：第2関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 11:  # 中指：第1関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 12:  # 中指：指先
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 13:  # 薬指：付け根
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 14:  # 薬指：第2関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 15:  # 薬指：第1関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 16:  # 薬指：指先
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 17:  # 小指：付け根
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 18:  # 小指：第2関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 19:  # 小指：第1関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 20:  # 小指：指先
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

        return image

    def draw_current_pointer_coordinates(self, image, point_history):
        if not point_history:
            return image  # If the history is empty, return the image as is

        # Assuming the last point in point_history is the current pointer
        current_point = point_history[-1]

        if current_point[0] == 0 and current_point[1] == 0:
            return image  # If the current point is (0,0), it's considered invalid/not present

        # Prepare the text to display the current point's coordinates
        coordinates_text = f"Current: ({current_point[0]}, {current_point[1]})"

        # Determine the position for the text (top right corner)
        text_size = cv.getTextSize(coordinates_text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = image.shape[1] - text_size[0] - 10  # 10 pixels from the right edge
        text_y = 20  # 20 pixels from the top

        # Draw a background rectangle for better readability
        cv.rectangle(
            image,
            (text_x - 5, text_y - 14),
            (text_x + text_size[0] + 5, text_y + 5),
            (0, 0, 0),
            -1,
        )

        # Display the coordinates text
        cv.putText(
            image,
            coordinates_text,
            (text_x, text_y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,  # Font scale
            (255, 255, 255),  # Font color
            1,  # Thickness
            cv.LINE_AA,
        )

        return image
