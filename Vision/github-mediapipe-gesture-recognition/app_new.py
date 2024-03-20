import csv
import copy
import argparse
import itertools
import json
import os
import socket
import time
import sys
import cv2 as cv
import numpy as np
import mediapipe as mp
from collections import Counter
from collections import deque
from datetime import datetime
from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier
from tcp.tcp_send import TCPClient

# Classes
from classes.Camera import Camera
from classes.MLModel import GestureModel
from classes.GestureProcessing import GestureProcessing
from classes.TCP import TCPCommunication


def main():
    # Initialize classes
    camera = Camera()
    gesture_model = GestureModel(camera)
    gesture_processing = GestureProcessing(gesture_model)
    tcp_communication = TCPCommunication(gesture_processing)

    while True:
        camera.calculate_fps()  # Calculate camera FPS, stored in camera.fps

        # Checks for 'ESC' and for gesture number-mode switching
        should_continue = camera.key_check()
        if not should_continue:  # If ESC is pressed, we terminate the program
            break

        camera.capture()  # Capture and pre-process a frame
        gesture_model.process_frame()  # Processes frame, results in hand landmarks


if __name__ == "__main__":
    main()
