import cv2
import numpy as np
import RPi.GPIO as GPIO
import time

# Initialize GPIO pins
GPIO.setmode(GPIO.BCM)
GPIO.setup(4, GPIO.OUT)
GPIO.setup(17, GPIO.OUT)

# Function to switch to camera A
def select_camera_a():
    GPIO.output(17, GPIO.LOW)
    GPIO.output(4, GPIO.LOW)
    # Additional setup if needed

# Function to switch to camera B
def select_camera_b():
    GPIO.output(17, GPIO.LOW)
    GPIO.output(4, GPIO.HIGH)
    # Additional setup if needed

# Initialize the camera
# Note: This assumes both cameras can be accessed via the same interface
# and that switching is handled externally (e.g., via a multiplexer)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame from camera A
    select_camera_a()
    time.sleep(0.1)  # Wait for the switch to take effect
    ret, frame_a = cap.read()

    # Capture frame-by-frame from camera B
    select_camera_b()
    time.sleep(0.1)  # Wait for the switch to take effect
    ret, frame_b = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Combine frames
    combined_frame = np.hstack((frame_a, frame_b))

    # Display the resulting frame
    cv2.imshow('frame', combined_frame)
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()
