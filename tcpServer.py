# tcpServer.py
# Receives video stream from Raspberry Pi
import socket
import cv2
import numpy as np


def startServer():
    serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serverAddr = ("0.0.0.0", 444)  # Binds to all interfaces
    serverSocket.bind(serverAddr)
    serverSocket.listen(1)

    print("Waiting for connection...")
    clientSocket, clientAddr = serverSocket.accept()
    print("Connection from ", clientAddr)

    try:
        # frame_counter = 0
        while True:
            # print(f"Receiving frame #{frame_counter}")
            # Read the length of the image data
            length = int.from_bytes(clientSocket.recv(4), byteorder="big")
            # print(f"Read length for frame #{frame_counter}")
            # Read the image data
            image_data = b""
            while len(image_data) < length:
                packet = clientSocket.recv(4096)
                if not packet:
                    break
                image_data += packet
            # print(f"Read image data for frame #{frame_counter}")
            # Convert the data to an image and display
            image = np.frombuffer(image_data, dtype=np.uint8)
            frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
            # print(f"Converted image data for frame #{frame_counter}")
            if frame is not None:
                cv2.imshow("Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            # frame_counter += 1

    finally:
        clientSocket.close()
        serverSocket.close()
        cv2.destroyAllWindows()


startServer()
