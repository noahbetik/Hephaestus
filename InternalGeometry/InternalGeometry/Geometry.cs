﻿// See https://aka.ms/new-console-template for more information
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;

Console.WriteLine("Hello, World! we r unga bunga ing");



Point testpoint = new Point(2.54, 9.71, 16.78);

Point p1 = new Point(1.2, 2.3, 3.4);
Point p2 = new Point(4.5, 5.6, 6.7);
Point p3 = new Point(7.8, 8.9, 9.10);

Spline testSpline = new Spline(p1, p2, [p3]);

double[] coords = testpoint.getCoords();
Console.WriteLine("New point object: {0}, {1}, {2}", coords[0], coords[1], coords[2]);

IPHostEntry ipHostInfo = await Dns.GetHostEntryAsync("localhost"); // get IP addresses
IPAddress ipAddress = ipHostInfo.AddressList[1]; // [1] is IPv4 addr from python server
IPEndPoint ipEndPoint = new(ipAddress, 444); // create endpoint with IP addr and port

using Socket client = new(ipEndPoint.AddressFamily, SocketType.Stream, ProtocolType.Tcp); // create socket client

await client.ConnectAsync(ipEndPoint); // wait for connection

while (true) // wait for message from Python-side TCP server
{
    /*// Send message.
    var message = "Hi friends 👋!<|EOM|>";
    var messageBytes = Encoding.UTF8.GetBytes(message);
    _ = await client.SendAsync(messageBytes, SocketFlags.None);
    Console.WriteLine($"Socket client sent message: \"{message}\"");*/

    var buffer = new byte[1024]; // read buffer
    var received = await client.ReceiveAsync(buffer, SocketFlags.None); // wait for message
    var msg = Encoding.UTF8.GetString(buffer, 0, received);
    Console.WriteLine($"Socket client received message: \"{msg}\"");
    if (msg == "end")
    {
        break;
    }

}

client.Shutdown(SocketShutdown.Both);


public class Point
{
    private double x;
    private double y;
    private double z;
    private readonly double limit = 100;

    public Point(double x, double y, double z)
    {
        // clip if out of range
        this.x = x < limit ? x : limit;
        this.y = y < limit ? y : limit;
        this.z = z < limit ? z : limit;
    }


    public void setCoords(double x, double y, double z)
    {
        // clip if out of range
        this.x = x < limit ? x : limit;
        this.y = y < limit ? y : limit;
        this.z = z < limit ? z : limit;
    }

    public double[] getCoords()
    {
        return [x, y, z];
    }
}


public abstract class Segment
{
    protected Point a;
    protected Point b;
    protected Point[]? c;

    protected Segment(Point a, Point b, Point[]? c = null)
    {
        this.a = a;
        this.b = b;
        this.c = c;
    }

    protected void setEndpoints(Point a, Point b)
    {
        this.a = a;
        this.b = b;
    }

    public Point[] getEndpoints()
    {
        return [a, b];
    }

    protected double vectorDistance(Point p1, Point p2)
    {
        double[] p1Coords = p1.getCoords();
        double[] p2Coords = p2.getCoords();
        return Math.Sqrt(Math.Pow(p2Coords[0] - p1Coords[0], 2) + Math.Pow(p2Coords[1] - p1Coords[1], 2) + Math.Pow(p2Coords[2] - p1Coords[2], 2));
    }

    protected Point segCentre(Point p1, Point p2)
    {
        double[] p1Coords = p1.getCoords();
        double[] p2Coords = p2.getCoords();
        return new Point((p1Coords[0] + p2Coords[0]) / 2, (p1Coords[1] + p2Coords[1]) / 2, (p1Coords[2] + p2Coords[2]) / 2);
    }

    public double getLength()
    {
        return vectorDistance(a, b);
    }

}

public class Line : Segment
{
    public Line(Point a, Point b) : base(a, b, null)
    {
        this.a = a;
        this.b = b;
    }
}

public class Spline : Segment
{
    public Spline(Point a, Point b, Point[] c) : base(a, b, c)
    {
        this.a = a;
        this.b = b;
    }

    public void setCurvepoints(Point[] c)
    {
        this.c = c;
    }

    public Point[] getCurvepoints()
    {
        return c!;
    }

    public new double getLength()
    {
        return 0; // MATH??
    }
}

public class Arc : Segment
{
    private double radius;
    private double angle;
    private double width;
    private double height;
    private Point? widthCentre;

    public Arc(Point a, Point b, Point c) : base(a, b, [c])
    {
        this.a = a; // endpoint 1
        this.b = b; // endpoint 2
        this.c = [c]; // midpoint

        recalculateGeometry();
    }

    private void recalculateGeometry()
    {
        this.width = vectorDistance(a, b); // necessary?
        this.widthCentre = segCentre(a, b);
        this.height = vectorDistance(widthCentre, c![0]);

        // radius = height/2 + (width^2 / 8*height)
        this.radius = (height / 2) + (Math.Pow(width, 2) / (8 * height));

        // cos(C) = (a^2 + b^2 - c^2) / (2ab)
        // a = b = radius
        // c = width
        this.angle = Math.Acos((Math.Pow(this.radius, 2) + Math.Pow(this.radius, 2) - Math.Pow(width, 2)) / 2 * Math.Pow(this.radius, 2));
    }

    public void setCurvepoint(Point c)
    {
        this.c = [c];
        recalculateGeometry();
    }

    public Point getCurvepoint()
    {
        return this.c![0];
    }

    public new double getLength()
    {
        return this.radius * this.angle;
    }

    public double getRadius()
    {
        return this.radius;
    }

    public double getAngle()
    {
        return this.angle;
    }
}

public abstract class Shape
{
    protected Point origin;
    protected List<Segment> segments = new List<Segment>();
    protected List<Point> points = new List<Point>();  // right now setting up for instantation either via points or segments -- should decide on one or the other eventually

    public Shape(Point origin, Segment[] segments)
    {
        this.origin = origin;
        this.segments = segments.ToList();
    }

    public Shape(Point origin, Point[] points)
    {
        this.origin = origin;
        this.points = points.ToList();
    }

    public void setOrigin(Point newOrigin)
    {
        this.origin = newOrigin;
    }

    public Point getOrigin()
    {
        return this.origin;
    }

}

public class Polygon : Shape
{
    public Polygon(Point origin, Segment[] segments) : base(origin, segments)
    {
        this.origin = origin;
        this.segments = segments.ToList();

        foreach (Segment sgmt in this.segments)
        {
            Point[] pt = sgmt.getEndpoints();
            foreach (Point p in pt)
            {
                if (!this.points.Contains(p))
                { // LEARN: C# pass by ref or pass by value? -- worth to check coord values regardless? -- TEST
                    this.points.Add(p);
                }
            }
        }
    }

    public Polygon(Point origin, Point[] points) : base(origin, points)
    {
        this.origin = origin;
        this.points = points.ToList();
        // TODO: generate Line[] from Point[]
        // TODO: need code to check if valid polygon (no intersecting lines)
        // if intersecting lines, create extra point for shared vertex
        // do we want functionality to split intersected polygons in to separate ones?

        Point startPoint;
        Point endPoint;
        Segment localSeg;

        for (int i = 0; i < this.points.Count - 1; i++)
        { // connect points in given order
            startPoint = this.points[i];
            endPoint = this.points[i + 1];
            localSeg = new Line(startPoint, endPoint);
            this.segments.Add(localSeg);
        }

        // connect last point with first point
        startPoint = this.points[this.points.Count - 1];
        endPoint = this.points[0];
        localSeg = new Line(startPoint, endPoint);
        this.segments.Add(localSeg);

        // intersection checking

    }


    public List<Segment> getSegments()
    {
        return this.segments;
    }

    public List<Point> getPoints()
    {
        return this.points;
    }

}

public abstract class Volume
{
    protected Point origin; // geomtric centre of base
    protected Shape base2d;
    protected Segment path;  // sweep path -- simple for extrusion, not sure how to handle revolution yet
    // TODO SILVER: deformations

    public Volume(Shape base2d, Segment path)
    {
        this.base2d = base2d;
        this.path = path;
        this.origin = base2d.getOrigin();
    }

    public void setOrigin(Point newOrigin)
    {
        this.origin = newOrigin;
    }

    public void setBase2d(Shape base2d)
    {
        this.base2d = base2d;
    }

    public void setPath(Segment path)
    {
        this.path = path;
    }

    public Point getOrigin()
    {
        return this.origin;
    }

    public Shape getBase2d()
    {
        return this.base2d;
    }

    public Segment getPath()
    {
        return this.path;
    }

}

public class Extrusion : Volume
{
    public Extrusion(Shape base2d, Segment path) : base(base2d, path)
    {
        this.base2d = base2d;
        this.path = path;
        this.origin = base2d.getOrigin();
    }

}