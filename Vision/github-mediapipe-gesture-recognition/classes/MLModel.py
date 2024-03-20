import mediapipe as mp
import csv
from collections import deque
from model import KeyPointClassifier
from model import PointHistoryClassifier


# Handles MediaPipe's Hand Model
class GestureModel:
    def __init__(self, camera):
        self.camera = camera
        self.initialize_model()
        self.initialize_classifiers()
        self.read_labels()
        self.initialize_histories()

    ## INITIALIZATIONS ########################
    # Initialize mediapipe's hand solution
    def initialize_model(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.camera.use_static_image_mode,  # treats each image as static, better to set as True if not a video stream
            max_num_hands=2,  # set 1 or 2 hands
            min_detection_confidence=self.camera.min_detection_confidence,  # minimum values, lower = easier to detect but more false positives
            min_tracking_confidence=self.camera.min_tracking_confidence,
        )

    # Initialize keypoint (hand gesture) and point history (pointer movement) classifiers
    def initialize_classifiers(self):
        self.keypoint_classifier = KeyPointClassifier()
        self.point_history_classifier = PointHistoryClassifier()

    # Read gesture labels for classification
    def read_labels(self):
        with open(
            "model/keypoint_classifier/keypoint_classifier_label.csv",
            encoding="utf-8-sig",
        ) as f:
            self.keypoint_classifier_labels = csv.reader(f)
            self.keypoint_classifier_labels = [
                row[0] for row in self.keypoint_classifier_labels
            ]
        with open(
            "model/point_history_classifier/point_history_classifier_label.csv",
            encoding="utf-8-sig",
        ) as f:
            self.point_history_classifier_labels = csv.reader(f)
            self.point_history_classifier_labels = [
                row[0] for row in self.point_history_classifier_labels
            ]

    # Initialize coordinate history
    def initialize_histories(self):
        # Coordinate history
        self.history_length = 16  # max length
        self.point_history = deque(maxlen=self.history_length)  # double ended queue
        self.finger_gesture_history = deque(maxlen=self.history_length)

    ## PROCESSING ########################
    def process_frame(self):
        self.results = self.hands.process(
            self.camera.image
        )  # mediapipe processing, results holds detected hand landmarks
        self.camera.image.flags.writeable = True
