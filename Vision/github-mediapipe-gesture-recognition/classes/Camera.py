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
        self.cap = cv.VideoCapture(self.cap_device)
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
