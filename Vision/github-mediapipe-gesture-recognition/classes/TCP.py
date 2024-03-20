import socket


# Facilitates TCP communications
class TCPCommunication:
    def __init__(self, gesture_processing):
        self.host = "localhost"
        self.port = 4446
        self.connect()
        self.gesture_processing = gesture_processing

    # Connect to specified host IP and port
    def connect(self):
        try:
            self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.connection.connect((self.host, self.port))
            print(f"Connected to {self.host}:{self.port}")
        except Exception as e:
            print(f"Failed to connect to {self.host}:{self.port}, {e}")
            self.connection = None

    # Send command
    def send_command(self, command):
        if self.connection:
            try:
                self.connection.sendall(command.encode(encoding="ascii"))
                print(f"Sent: {command}")
            except Exception as e:
                print(f"Failed to send data: {e}")
                # Attempt to reconnect if sending fails
                self.connect()
        else:
            print("No connection established. Trying to reconnect...")
            self.connect()

    # Close TCP connection
    def close(self):
        if self.connection:
            self.connection.close()
            print("Connection closed.")
