import cv2

def main():
    print("Starting camera capture...")
    cap= cv2.VideoCapture(0,cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    else:
        print("Camera opened successfully.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        cv2.imshow('Camera Feed', frame)
        print("Showing camera feed...")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera capture stopped.")

if __name__ == "__main__":
    main()
