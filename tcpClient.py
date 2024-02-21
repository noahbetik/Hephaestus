# tcpClient.py

import socket as s

def startClient():
    clientSocket = s.socket(s.AF_INET, s.SOCK_STREAM)
    serverAddr = ('localhost', 444) # (IP addr, port)
    clientSocket.connect(serverAddr)
    while True:
        data = clientSocket.recv(1024)
        print("Received: ", data.decode())

    clientSocket.close()
    
startClient()
