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
                ready = select.select([self.connection], [], [], 0.1)  # Adjusted timeout for demonstration
                if ready[0]:
                    data = self.connection.recv(1024).decode("ascii")
                    #print("received ", data)
                    command_response = data.split(" ")[0].strip()
                    
                    # Default to acknowledgment unless proven otherwise
                    self.last_command_acknowledged = True
                    
                    if "RST" in command_response:    
                        self.rst = 1
                    #    print("Received RST packet.")
                    elif "SEL" in command_response:                       
                        self.sel = 1
                        self.desel = 0
                       # print("Received SEL packet.")
                    elif "DES" in command_response:    
                        self.desel = 1
                        self.sel = 0
                       # print("Received DES packet.")
                    else:
                        # Only in this case did we not receive a known response
                        self.last_command_acknowledged = False
                        print(f"Received unexpected response: {data}")
                else:
                    #print("Acknowledgment not received within timeout.")
                    self.last_command_acknowledged = False
                        
                if not self.last_command_acknowledged:
                    #print("Last command not acknowledged. Not sending new command.")
                    return
                else:
                    self.connection.sendall(command.encode("ascii"))
                    #print(f"Sent: {command.strip()}")

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
