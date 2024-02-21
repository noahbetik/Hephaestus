using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Text;

public class UnityTCPClient : MonoBehaviour
{
    public static UnityTCPClient instance;
    public static int dataBufferSize = 4096;

    public string ipAddr = "127.0.0.1";
    public int port = 444;
    public int myId = 0;
    public TCP tcp;

    private void Awake() // ensure only once instance running
    {
        if (instance == null)
        {
            instance = this;
        }
        else if (instance != this)
        {
            Console.WriteLine("destroying duplicate instance.");
            Destroy(this);
        }
    }

    private void Start()
    {
        tcp = new TCP();
    }

    public void ConnectToServer()
    {

    }

    public class TCP
    {
        public TcpClient socket;
        private NetworkStream stream;
        private byte[] recvBuf;

        public void Connect()
        {
            socket = new TcpClient
            {
                socket.ReceiveBufferSize = dataBufferSize,
                socket.SendBufferSize = dataBufferSize
            };

            recvBuf = new byte[dataBufferSize];
            socket.BeginConnect(instance.ipAddr, instance.port, ConnectCallback, socket);
        }

        private void ConnectCallback(IAsyncResult ar)
        {
            socket.EndConnect(ar);

            if (!socket.Connected)
            {
                return;
            }

            stream = socket.GetStream();
            stream.BeginRead(recvBuf, 0, dataBufferSize, ReceiveCallback, null);
        }

        private void ReceiveCallback(IAsyncResult ar)
        {
            try
            {
                int byteLength = stream.EndRead(ar);
                if (byteLength > 0)
                {
                    // TODO: disconnect ??
                    return;
                }

                byte[] data = new byte[byteLength];
                Array.Copy(recvBuf, data, byteLength);

                // TODO: handle data
                stream.BeginRead(recvBuf, 0, dataBufferSize, ReceiveCallback, null);

            }
            catch (Exception e)
            {
                Console.WriteLine("there's been a problem:\n");
                Console.WriteLine(e.ToString());
                // TODO: disconnect
            }
        }

    }

}
