import socket


def start_server(host="127.0.0.1", port=4449):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print(f"Server listening on {host}:{port}")
        while True:  # Keep the server running
            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr}")
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break  # Exit the loop if client closes or data is not received
                    print(f"Received: {data.decode()}")
                print("Client disconnected, waiting for new connection...")


if __name__ == "__main__":
    start_server()
