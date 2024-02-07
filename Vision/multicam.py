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


/home/capstone/RaspberryPi/Multi_Camera_Adapter/Multi_Adapter_Board_2Channel_uc444/multicam.py:8: RuntimeWarning: This channel is already in use, continuing anyway.  Use GPIO.setwarnings(False) to disable warnings.
  GPIO.setup(4, GPIO.OUT)
/home/capstone/RaspberryPi/Multi_Camera_Adapter/Multi_Adapter_Board_2Channel_uc444/multicam.py:9: RuntimeWarning: This channel is already in use, continuing anyway.  Use GPIO.setwarnings(False) to disable warnings.
  GPIO.setup(17, GPIO.OUT)
[ WARN:0@0.430] global ./modules/videoio/src/cap_gstreamer.cpp (2401) handleMessage OpenCV | GStreamer warning: Embedded video playback halted; module v4l2src0 reported: Failed to allocate required memory.
[ WARN:0@0.431] global ./modules/videoio/src/cap_gstreamer.cpp (1356) open OpenCV | GStreamer warning: unable to start pipeline
[ WARN:0@0.431] global ./modules/videoio/src/cap_gstreamer.cpp (862) isPipelinePlaying OpenCV | GStreamer warning: GStreamer: pipeline have not been created
Can't receive frame (stream end?). Exiting ...

