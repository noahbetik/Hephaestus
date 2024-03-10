# tcpServer.py

import socket as s


def startServer():
    serverSocket = s.socket(s.AF_INET, s.SOCK_STREAM)
    serverAddr = ("localhost", 444)  # (IP addr, port)
    serverSocket.bind(serverAddr)
    serverSocket.listen(1)  # max number of queued connections

    print("Waiting for connection...")
    clientSocket, clientAddr = serverSocket.accept()
    print("Connection from ", clientAddr)

    while True:
        data = input()
        clientSocket.sendall(data.encode())
        if data == "end":
            break

    clientSocket.close()
    serverSocket.close()


startServer()
