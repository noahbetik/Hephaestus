import socket
import select

class TCPCommunication:
    def __init__(self, gesture_processing):
        self.host = "localhost"
        self.port = 4445
        self.gesture_processing = gesture_processing
        self.connect()

    def connect(self):
        try:
            self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.connection.connect((self.host, self.port))
            print(f"\nConnected to {self.host}:{self.port}")
        except Exception as e:
            print(f"Failed to connect to {self.host}:{self.port}, {e}")
            self.connection = None

    def send_command(self, command):
        command = command + "\n"
        if self.connection:
            try:
                self.connection.sendall(command.encode(encoding="ascii"))
                print(f"Sent: {command.strip()}")

                # Wait for acknowledgment
                # ready = select.select([self.connection], [], [], 5)  # 5 second timeout
                # if ready[0]:
                #     data = self.connection.recv(1024).decode("ascii")
                #     if data.strip() == "ACK":
                #         print("Acknowledgment received.")
                #     else:
                #         print("Failed to receive acknowledgment. Discarding packet.")
                # else:
                #     print("Acknowledgment not received within timeout.")

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
            print("\nConnection closed.")
