// See https://aka.ms/new-console-template for more information
using System.Collections.Generic;

Console.WriteLine("Hello, World! we r unga bunga ing");

Point testpoint = new Point(2.54, 9.71, 16.78);

Point p1 = new Point(1.2, 2.3, 3.4);
Point p2 = new Point(4.5, 5.6, 6.7);
Point p3 = new Point(7.8, 8.9, 9.10);

Spline testSpline = new Spline(p1, p2, [p3]);

double[] coords = testpoint.getCoords();
Console.WriteLine("New point object: {0}, {1}, {2}", coords[0], coords[1], coords[2]);


public class Point
{
    private double x;
    private double y;
    private double z;
    private double limit = 100;

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


public class Segment
{
    protected Point a;
    protected Point b;
    protected Point[]? c;

    public Segment(Point a, Point b, Point[]? c = null)
    {
        this.a = a;
        this.b = b;
        this.c = c;
    }

    public void setEndpoints(Point a, Point b)
    {
        this.a = a;
        this.b = b;
    }

    public Point[] getendPoints()
    {
        return [a, b];
    }

    public double getLength()
    {
        double[] aCoords = a.getCoords();
        double[] bCoords = b.getCoords();
        return Math.Sqrt(Math.Pow(bCoords[0] - aCoords[0], 2) + Math.Pow(bCoords[1] - aCoords[1], 2) + Math.Pow(bCoords[2] - aCoords[2], 2));
    }

}

public class Line : Segment {
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

    public void setCurvepoints(Point[] c) {
        this.c = c;
    }

    public Point[] getCurvepoints() {
        return c;
    }

    public new double getLength() {
        return 0; // MATH??
    }
}

public class Arc : Segment
{
    public Arc(Point a, Point b, Point c) : base(a, b, [c])
    {
        this.a = a;
        this.b = b;
    }

    public void setCurvepoint(Point c)
    {
        this.c = [c];
    }

    public Point[] getCurvepoint()
    {
        return c;
    }

    public new double getLength()
    {
        return 0; // MATH??
    }

    public double getRadius()
    {
        return 0; // MATH??
    }

    public double getAngle()
    {
        return 0; // MATH??
    }
}