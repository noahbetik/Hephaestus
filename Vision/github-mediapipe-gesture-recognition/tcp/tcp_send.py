import socket


class TCPClient:
    def __init__(self, host="127.0.0.1", port=65432):
        self.host = host
        self.port = port
        self.connection = None

    def connect(self):
        try:
            self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.connection.connect((self.host, self.port))
            print(f"Connected to {self.host}:{self.port}")
        except Exception as e:
            print(f"Failed to connect to {self.host}:{self.port}, {e}")
            self.connection = None

    def send_gesture(self, gesture_name):
        if self.connection:
            try:
                self.connection.sendall(gesture_name.encode())
                print(f"Sent: {gesture_name}")
            except Exception as e:
                print(f"Failed to send data: {e}")
                # Attempt to reconnect if sending fails
                self.connect()
        else:
            print("No connection established. Trying to reconnect...")
            self.connect()

    def close(self):
        if self.connection:
            self.connection.close()
            print("Connection closed.")
