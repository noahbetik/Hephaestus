# socket yeet demo

import socket as s

def startServer():
    serverSocket = s.socket(s.AF_INET, s.SOCK_STREAM)
    serverAddr = ('localhost', 444) # (IP addr, port)
    serverSocket.bind(serverAddr)
    serverSocket.listen(1) # max number of queued connections

    print("Waiting for connection...")
    clientSocket, clientAddr = serverSocket.accept()
    print("Connection from ", clientAddr)

    l = ["(0.4,0.2,0)", "(0.6,0.3,0)", "(0.8,0.2,0)", "(1,0.1,0)", "(0.8,0,0)", "(0.6,-0.1,0)", "(0.4,-0.2,0)"]

    clientSocket.send("create line start (0,0) (0.2,0.1)".encode())

    for item in l:
        clientSocket.send(l.encode())

    clientSocket.send("create line end (0.2,-0.1)")

    clientSocket.close()
    serverSocket.close()
    
startServer()
