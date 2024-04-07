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
from classes.StateMachine import StateMachine


# List of all gesture indexes
gesture_list = {
    0: "4 Fingers",  # toggle, one-hit
    1: "Pointer",
    2: "2 Fingers",
    3: "Rock on",
    4: "3 Fingers",
    5: "Thumbs Down",  # one-hit
    6: "Fist",  # dual only
    7: "L_RightHand",  # dual only, one-hit
    8: "L_LeftHand",  # dual only, one-hit
    9: "Thumbs Up",  # one-hit
    10: "Illuminati_LeftHand",  # dual only, one-hit
    11: "Illuminati_RightHand",  # dual only, one-hit
    12: "Extrude",
    13: "The Gun",
    (1, 6): "One finger and a fist",
    (2, 6): "Two fingers and a fist",
    (4, 6): "Three fingers and a fist",
    (1, 3): "Add CUBE",
    (2, 3): "Add SPHERE",
    (4, 3): "Add TRIANGLE",
    (10, 11): "Illuminati",
    (8, 7): "L Shape",
}


def start_command(
    gesture_type,
    gesture_subtype,
    point_history,
    axis="z",
    snap_view="iso",
    object="cube",
):
    match (gesture_type, gesture_subtype):
        case ("motion", "zoom"):  # 2 fingers
            return f"motion zoom start {point_history[-1][0]}"
        case ("motion", "pan"):  # 3 fingers
            return f"motion pan start ({point_history[-1][0]},{point_history[-1][1]})"
        case ("motion", "rotate"):  # fist
            return f"motion rotate start {axis} {point_history[-1][0]}"
        case ("motion", "extrude"):  # pinch and drag
            return (
                f"motion extrude start ({point_history[-1][0]},{point_history[-1][1]})"
            )
        case ("create", "line"):  # pointer
            return f"create line start ({point_history[-2][0]},{point_history[-2][1]}) ({point_history[-1][0]},{point_history[-1][1]})"  # from noah: i do it this way to make parsing easier on UI side
        case ("toggle", "mode"):
            return "toggle mode"
        case ("toggle", "motion"):
            return "toggle motion"
        case ("snap", "None"):
            return f"snap {snap_view}"
        case ("select", "None"):
            return f"select"
        case ("deselect", "None"):
            return f"deselect"
        case ("delete", "None"):
            return f"delete"
        case ("create", "object"):
            return f"create object {object}"
        case _:
            return "Command not found"


def active_command(gesture_type, gesture_subtype, point_history):
    match (gesture_type, gesture_subtype):
        case ("motion", "zoom"):
            return f"motion zoom position {point_history[-1][0]}"
        case ("motion", "pan"):
            return (
                f"motion pan position ({point_history[-1][0]},{point_history[-1][1]})"
            )
        case ("motion", "rotate"):
            return f"motion rotate position {point_history[-1][0]}"
        case ("motion", "extrude"):
            return f"motion extrude position ({point_history[-1][0]},{point_history[-1][1]})"
        case ("create", "line"):
            return f"create line ({point_history[-1][0]},{point_history[-1][1]})"
        case ("toggle", "mode"):
            return "toggle mode"
        case ("toggle", "motion"):
            return ""
        case ("snap", "None"):
            return ""
        case ("snap_righthand", "None"):
            return ""
        case ("snap_lefthand", "None"):
            return ""
        case ("snap_iso", "left"):
            return ""
        case ("snap_iso", "right"):
            return ""
        case _:
            return "Command not found"


def load_gesture_definitions(filename):
    with open(filename, "r") as file:
        gesture_definitions = json.load(file)
    return gesture_definitions


def main():
    left_hand_gesture_id = None
    right_hand_gesture_id = None
    confidence_threshold = 0.50
    frame_threshold = 8
    frame_counter = 0
    # 0 = nothing
    # 1 = lock-in
    # 2 = active
    # 3 = end
    state_machine = 0
    left_hand_gesture_id = None
    right_hand_gesture_id = None
    prev_num_of_hands = 0
    prev_left_hand_gesture_id = None
    prev_right_hand_gesture_id = None
    axis = "x"
    view = "home"

    gesture_types = load_gesture_definitions("./tcp/gestures.json")

    # Initialize classes
    camera = Camera()
    gesture_model = GestureModel(camera)
    gesture_processing = GestureProcessing(gesture_model)
    tcp_communication = TCPCommunication(gesture_processing)
    # state_machine = StateMachine()
    while True:
        try:
            camera.calculate_fps()  # Calculate camera FPS, stored in camera.fps

            # Checks for 'ESC' and for gesture number-mode switching
            should_continue = camera.key_check()
            if not should_continue:  # If ESC is pressed, we terminate the program
                break

            try:
                camera.capture()  # Capture and pre-process a frame
            except Exception as e:
                print("Camera capture failed! Continuing")
                print(e)
                continue
            gesture_model.process_frame()  # Processes frame, results in hand landmarks

            # If there is a hand in the view
            if gesture_model.results.multi_hand_landmarks is not None:

                # Get # of hands
                num_of_hands = 1
                temp_num_of_hands = len(gesture_model.results.multi_handedness)

                # Process landmarks
                for hand_landmarks, handedness in zip(
                    gesture_model.results.multi_hand_landmarks,
                    gesture_model.results.multi_handedness,
                ):
                    # Process the landmark list
                    gesture_model.process_landmark_list(
                        hand_landmarks, handedness, camera
                    )

                    # Classify the landmark list
                    hand_sign_id, confidence = gesture_model.keypoint_classifier(
                        gesture_model.pre_processed_landmark_list
                    )

                    # Assign the gesture ID based on hand label
                    if gesture_model.hand_label == "Left":
                        left_hand_gesture_id = hand_sign_id
                    elif gesture_model.hand_label == "Right":
                        right_hand_gesture_id = hand_sign_id
                        gesture_model.point_history.append(
                            gesture_model.landmark_list[8]
                        )

                if confidence > confidence_threshold:
                    num_of_hands = temp_num_of_hands
                else:
                    print(f"\nConfidence too low: {confidence}")
                    cv.imshow("Hand Gesture Recognition", camera.debug_image)
                    continue

                # What stage are we in?
                ## NOTHING STAGE ##############################################################
                if state_machine == 0:
                    single_gesture_name = None
                    dual_gesture_name = None
                    # How many hands are there?
                    # Single gesture
                    if num_of_hands == 1 and (
                        hand_sign_id != 0
                        and hand_sign_id != 6
                        and hand_sign_id != 7
                        and hand_sign_id != 8
                        and hand_sign_id != 10
                        and hand_sign_id != 11
                    ):
                        # Is the single gesture a valid gesture?
                        single_gesture_name = gesture_list[hand_sign_id]
                        if single_gesture_name:
                            state_machine = 1
                            print(
                                f"\nGesture detected as single: {single_gesture_name}. Going to locking-in stage"
                            )
                        else:
                            print(
                                f"\nFor some reason, the single gesture couldn't be grabbed from the gestures list."
                            )
                    # Dual gesture
                    elif num_of_hands == 2:
                        # Is the dual gesture a valid gesture?
                        try:
                            dual_gesture_name = gesture_list[
                                (left_hand_gesture_id, right_hand_gesture_id)
                            ]
                            state_machine = 1
                            print(
                                f"\nGesture detected as dual: {dual_gesture_name}. Going to locking-in stage"
                            )
                        except:
                            print("\nNot a valid dual gesture.")

                ## LOCKING-IN STAGE ##############################################################
                elif state_machine == 1:  # Lock-in
                    # Was a change made? (Changed # of hands, or changed gesture)
                    if confidence > confidence_threshold:
                        if frame_counter > 0:
                            gesture_changed = (
                                left_hand_gesture_id != prev_left_hand_gesture_id
                                or right_hand_gesture_id != prev_right_hand_gesture_id
                            )
                            num_hands_changed = num_of_hands != prev_num_of_hands
                            # print(
                            #     f"num_hands = {num_of_hands}, prev_num_hands = {prev_num_of_hands}"
                            # )

                            if gesture_changed or num_hands_changed:
                                frame_counter = 0
                                state_machine = 0
                                print(
                                    f"\nGESTURE_CHANGED = {gesture_changed}, num_hands_changed = {num_hands_changed}, MOVING TO NOTHING STAGE. "
                                )
                                continue
                        frame_counter += 1

                        # Set gesture_name
                        gesture_name = (
                            dual_gesture_name
                            if num_of_hands == 2 and dual_gesture_name is not None
                            else single_gesture_name
                        )
                        if gesture_name == "Thumbs Up" or gesture_name == "Thumbs Down":
                            frame_threshold = 5

                        sys.stdout.write(
                            f"\rGesture: {gesture_name}, Frames: {frame_counter}, Confidence: {confidence:.5f}"
                        )
                        sys.stdout.flush()
                        if frame_counter >= frame_threshold:
                            frame_counter = 0
                            gesture_type = gesture_types[
                                gesture_list[right_hand_gesture_id]
                            ]["type"]
                            gesture_subtype = gesture_types[
                                gesture_list[right_hand_gesture_id]
                            ]["subtype"]

                            match left_hand_gesture_id:
                                case 1:  # 1 finger
                                    axis = "x"
                                    object = "cube"
                                case 2:  # 2 fingers
                                    axis = "y"
                                    object = "sphere"
                                case 4:  # 3 fingers
                                    axis = "z"
                                    object = "triangle"
                                case 8:  # L_LeftHand, home
                                    view = "home"
                                case 10:  # Illuminati_LeftHand, iso
                                    view = "iso"

                            gesture_start_command = start_command(
                                gesture_type,
                                gesture_subtype,
                                gesture_model.point_history,
                                axis,
                                view,
                                object,
                            )
                            print("\n")
                            print(gesture_start_command)
                            tcp_communication.send_command(gesture_start_command)
                            if (
                                gesture_types[gesture_list[right_hand_gesture_id]][
                                    "action"
                                ]
                                == "one-hit"
                            ):
                                state_machine = 0
                                print("\nOne-hit, transitioning to NOTHING stage\n")
                            else:
                                state_machine = 2
                                print("\nTransitioning to ACTIVE stage\n")

                        prev_num_of_hands = num_of_hands
                        prev_left_hand_gesture_id = left_hand_gesture_id
                        prev_right_hand_gesture_id = right_hand_gesture_id
                    else:
                        print(f"\nConfidence too low: {confidence}")
                ## ACTIVE STAGE ###############################################################
                elif state_machine == 2:  # Active
                    gesture_changed = (
                        left_hand_gesture_id != prev_left_hand_gesture_id
                        or right_hand_gesture_id != prev_right_hand_gesture_id
                    )
                    if left_hand_gesture_id == 5 or right_hand_gesture_id == 5:
                        state_machine = 3
                        print("Terminating gesture due to thumbs-down")
                        continue
                    gesture_active_command = active_command(
                        gesture_type, gesture_subtype, gesture_model.point_history
                    )
                    tcp_communication.send_command(gesture_active_command)
                    # print(gesture_active_command)
                ## END STAGE ##################################################################
                elif state_machine == 3:  # End
                    state_machine = 0
                    gesture_end_command = f"{gesture_type} {gesture_subtype} end"
                    tcp_communication.send_command(gesture_end_command)

                camera.draw_bounding_rect(gesture_model.brect)
                camera.debug_image = camera.draw_landmarks(
                    camera.debug_image, gesture_model.landmark_list
                )
                camera.debug_image = camera.draw_info_text(
                    camera.debug_image,
                    gesture_model.brect,
                    handedness,
                    gesture_model.keypoint_classifier_labels[hand_sign_id],
                    "",
                )
                camera.debug_image = camera.draw_current_pointer_coordinates(
                    camera.debug_image, gesture_model.point_history
                )
            ## NOTHING STAGE ###################################
            else:
                if right_hand_gesture_id:
                    if state_machine != 0 and not (
                        gesture_types[gesture_list[right_hand_gesture_id]]["action"]
                        == "one-hit"
                    ):
                        state_machine = 0
                        gesture_end_command = f"{gesture_type} {gesture_subtype} end"
                        tcp_communication.send_command(gesture_end_command)
                # print(gesture_end_command)
                prev_left_hand_gesture_id = None
                prev_right_hand_gesture_id = None
                left_hand_gesture_id = None
                right_hand_gesture_id = None

                sys.stdout.write(f"\rThere is no hand in view")
                sys.stdout.flush()

            cv.imshow("Hand Gesture Recognition", camera.debug_image)
            left_hand_gesture_id = None
            right_hand_gesture_id = None

        except Exception as e:
            print(f"\nException, continuing: {e}.\nTry your right hand instead?")
            continue

    tcp_communication.close()
    camera.cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
