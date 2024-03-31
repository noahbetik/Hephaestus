import mediapipe as mp
import csv
import cv2 as cv
import numpy as np
import itertools
import copy
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

    def process_landmark_list(self, hand_landmarks, handedness, camera):
        self.hand_label = handedness.classification[0].label  # Right or left
        # Bounding box calculation
        self.brect = self.calc_bounding_rect(camera.debug_image, hand_landmarks)
        # Landmark calculation
        self.landmark_list = self.calc_landmark_list(camera.debug_image, hand_landmarks)

        # Conversion to relative coordinates / normalized coordinates
        self.pre_processed_landmark_list = self.pre_process_landmark(self.landmark_list)
        self.pre_processed_point_history_list = self.pre_process_point_history(
            camera.debug_image, self.point_history
        )

    # OTHER
    def calc_bounding_rect(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_array = np.empty((0, 2), int)

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point = [np.array((landmark_x, landmark_y))]

            landmark_array = np.append(landmark_array, landmark_point, axis=0)

        x, y, w, h = cv.boundingRect(landmark_array)

        return [x, y, x + w, y + h]

    def calc_landmark_list(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        # Keypoint
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            # landmark_z = landmark.z

            landmark_point.append([landmark_x, landmark_y])

        return landmark_point

    def pre_process_landmark(self, landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        # Convert to a one-dimensional list
        temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

        # Normalization
        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))

        return temp_landmark_list

    def pre_process_point_history(self, image, point_history):
        image_width, image_height = image.shape[1], image.shape[0]

        temp_point_history = copy.deepcopy(point_history)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, point in enumerate(temp_point_history):
            if index == 0:
                base_x, base_y = point[0], point[1]

            temp_point_history[index][0] = (
                temp_point_history[index][0] - base_x
            ) / image_width
            temp_point_history[index][1] = (
                temp_point_history[index][1] - base_y
            ) / image_height

        # Convert to a one-dimensional list
        temp_point_history = list(itertools.chain.from_iterable(temp_point_history))

        return temp_point_history
