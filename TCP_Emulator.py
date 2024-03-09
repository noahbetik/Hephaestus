import socket as s
import sys
import time

def startServer():
    serverSocket = s.socket(s.AF_INET, s.SOCK_STREAM)
    serverAddr = ('localhost', 444) # (IP addr, port)
    serverSocket.bind(serverAddr)
    return serverSocket

def runServer(serverSocket):
    serverSocket.listen(1) # max number of queued connections

    print("Waiting for connection...")
    clientSocket, clientAddr = serverSocket.accept()
    print("Connection from ", clientAddr)

    with open('demo_cmd_set.txt', 'r') as f:
        for line in f:
            if (line == "finish"):
                return 0
            print(line.strip("\n"))
            time.sleep(0.01)
            clientSocket.send(line.strip("\n").encode(encoding="ascii"))

    clientSocket.close()


sock = startServer()

loop = 1

while(loop):
    loop = runServer(sock)
