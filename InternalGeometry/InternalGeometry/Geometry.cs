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

    protected Point[] getendPoints()
    {
        return [a, b];
    }

    protected double vectorDistance(Point p1, Point p2) {
        double[] p1Coords = p1.getCoords();
        double[] p2Coords = p2.getCoords();
        return Math.Sqrt(Math.Pow(p2Coords[0] - p1Coords[0], 2) + Math.Pow(p2Coords[1] - p1Coords[1], 2) + Math.Pow(p2Coords[2] - p1Coords[2], 2));
    }

    protected Point segCentre(Point p1, Point p2) {
        double[] p1Coords = p1.getCoords();
        double[] p2Coords = p2.getCoords();
        return new Point((p1Coords[0] + p2Coords[0])/2, (p1Coords[1] + p2Coords[1])/2, (p1Coords[2] + p2Coords[2])/2);
    }

    protected double getLength()
    {
        return vectorDistance(a, b);
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
        return c!;
    }

    public new double getLength() {
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

    private void recalculateGeometry() {
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
    protected Segment[] segments = [];
    protected Point[] points = [];  // right now setting up for instantation either via points or segments -- should decide on one or the other eventually

    public Shape(Point origin, Segment[] segments) {
        this.origin = origin;
        this.segments = segments;
    }

    public Shape(Point origin, Point[] points)
    {
        this.origin = origin;
        this.points = points;
    }

}

public class Polygon : Shape {

    // LEARN: segments is <Segment> in base class but is passed as <Line> to Polygon constructor
    // which obj methods are available? Segment or Line?

    public Polygon(Point origin, Line[] segments) : base(origin, segments){
        this.origin = origin;
        this.segments = segments;
        // TODO: generate Point[] from line endpoints
    }

    public Polygon(Point origin, Point[] points) : base(origin, points)
    {
        this.origin = origin;
        this.points = points;
        // TODO: generate Line[] from Point[]
        // TODO: need code to check if valid polygon (no intersecting lines)
        // if intersecting lines, create multiple Polygons sharing vertices
    }

}