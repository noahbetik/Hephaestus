import socket
import select

class TCPCommunication:
    def __init__(self, gesture_processing):
        self.host = "localhost"
        self.port = 4445
        self.gesture_processing = gesture_processing
        self.sel =""
        self.desel = ""
        self.rst = ""
        self.last_command_acknowledged = True  # Initial state allows sending the first command
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
 

        command += "\n"
        if self.connection:
            try:
    

                # Wait for acknowledgment
                ready = select.select([self.connection], [], [], 0.1)  # 5 second timeout
                if ready[0]:
                    data = self.connection.recv(1024).decode("ascii")
                    print("received ", data)
                    command_response = data.split(" ")[0].strip()
                    
                    if command_response == "ACK":
                        print("Acknowledgment received.")
                        self.last_command_acknowledged = True
                    elif command_response == "SEL":
                        self.sel = 1
                        self.desel = 0
                        self.last_command_acknowledged = True  # Assuming SEL counts as acknowledgment
                        print("Received SEL packet.")
                    elif command_response == "DESEL":
                        self.desel = 1
                        self.sel = 0
                        self.last_command_acknowledged = True  # Assuming DESEL counts as acknowledgment
                        print("Received DESEL packet.")
                    elif command_response == "RST":
                        self.rst = 1
                        self.last_command_acknowledged = True  # Assuming RST counts as acknowledgment
                        print("Received RST packet.")
                    else:
                        self.last_command_acknowledged = False
                        print(f"Received unexpected response: {data}")
                else:
                    print("Acknowledgment not received within timeout.")
                    self.last_command_acknowledged = False
                    
                    
                if not self.last_command_acknowledged:
                    print("Last command not acknowledged. Not sending new command.")
                    return
                else:
                    self.connection.sendall(command.encode("ascii"))
                    print(f"Sent: {command.strip()}")

            except Exception as e:
                print(f"Failed to send data: {e}")
                self.last_command_acknowledged = False
                # Attempt to reconnect if sending fails
                self.connect()
        else:
            print("No connection established. Trying to reconnect...")
            self.connect()
    def close(self):
        if self.connection:
            self.connection.close()
            print("\nConnection closed.")
