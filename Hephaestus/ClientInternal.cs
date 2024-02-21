using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;

public class ClientInternal
{
    public static int dataBufferSize = 4096;
    public int id;
    public TCPClient tcp;

    public class TCP
    {
        public TcpClient socket;
        private readonly int id;
        private NetworkStream stream;
        private byte[] recvBuf;


        public TCPClient(int id)
        {
            this.id = id;
        }

        public void Connect(TCPClient socket)
        {
            this.socket = socket;
            socket.ReceiveBufferSize = dataBufferSize;
            socket.SendBufferSize = dataBufferSize;

            this.stream = socket.GetStream();
            recvBuf = new byte[dataBufferSize];
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


