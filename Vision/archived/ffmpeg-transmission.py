import cv2

# Replace with the same PORT used in the ffmpeg command
stream_url = 'udp://@:444'

cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Error opening video stream")
else:
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Process the frame with OpenCV or MediaPipe here

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

cap.release()
cv2.destroyAllWindows()
