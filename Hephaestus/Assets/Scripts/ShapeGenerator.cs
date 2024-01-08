using System.Collections.Generic;
using UnityEngine;

//notes
//set of global coordinates as a 'home'
//


public class ShapeGenerator : MonoBehaviour
{
    [SerializeField] private Material shapeMaterial; // Assign this in the inspector
    [SerializeField] private Material diamondMaterial; 

    // Start is called before the first frame update
    void Start()
    {
        //Vector3 p1 = new Vector3(1.2f, 2.3f, 3.4f);
        //Vector3 p2 = new Vector3(4.5f, 5.6f, 6.7f);

        //GenerateLine(p1, p2);
        //CreateSquarePlane();
        //CreateDiamond();

        Mesh diamondMesh = CreateDiamond();
        // Choose the face to extrude. For example, the bottom face of the diamond
        int[] faceToExtrude = { 5, 2, 1 }; // Indices of the bottom face vertices
        float extrusionDistance = 10.0f; // Define how much you want to extrude
        ExtrudeFace(diamondMesh, faceToExtrude, extrusionDistance);  
    }

    // Generate a line in Unity's 3D space using LineRenderer
    void GenerateLine(Vector3 start, Vector3 end)
    {
        GameObject lineObject = new GameObject("Line");
        LineRenderer lineRenderer = lineObject.AddComponent<LineRenderer>();

        // Set the material to the line
        lineRenderer.material = shapeMaterial;

        // Set the width of the LineRenderer
        lineRenderer.startWidth = 0.1f;
        lineRenderer.endWidth = 0.1f;

        // Set the number of points to the line
        lineRenderer.positionCount = 2;

        // Set the line's points
        lineRenderer.SetPosition(0, start);
        lineRenderer.SetPosition(1, end);
    }

    void CreateSquarePlane()

    {

        Debug.Log("Creating square plane");
        // Define the vertices of the square plane
        Vector3[] vertices = {
            new Vector3(-2, 0, -2),
            new Vector3(2, 0, -2),
            new Vector3(2, 0, 2),
            new Vector3(-2, 0, 2)
        };

        // Define the triangles that make up the square's two faces
        int[] triangles = {
            0, 1, 2,
            2, 3, 0
        };

        // Create a new Mesh and populate with data
        Mesh mesh = new Mesh();
        mesh.vertices = vertices;
        mesh.triangles = triangles;
        mesh.RecalculateNormals(); // Automatically calculate normals for lighting

        // Set up game object with mesh;
        GameObject squarePlane = new GameObject("SquarePlane", typeof(MeshFilter), typeof(MeshRenderer));

        // Assign the mesh to the MeshFilter
        squarePlane.GetComponent<MeshFilter>().mesh = mesh;

        // Assign material to the MeshRenderer
        squarePlane.GetComponent<MeshRenderer>().material = shapeMaterial;

        MeshCollider meshCollider = squarePlane.AddComponent<MeshCollider>();
        meshCollider.sharedMesh = mesh; // Use the same mesh for the collider
    }


    Mesh CreateDiamond()
    {
        // Define the vertices of the diamond
        Vector3[] vertices = new Vector3[]
        {
            new Vector3(0, 1, 0),    // Top point (tip)
            new Vector3(-1, 0, -1),  // Base vertices
            new Vector3(1, 0, -1),
            new Vector3(1, 0, 1),
            new Vector3(-1, 0, 1),
            new Vector3(0, -1, 0)    // Bottom point (tip)
        };

        // Define the triangles that make up the diamond's faces
        int[] triangles = new int[]
        {
            // Top half (four triangular sides)
            0, 1, 2,
            0, 2, 3,
            0, 3, 4,
            0, 4, 1,
            // Bottom half (four triangular sides)
            5, 2, 1,
            5, 3, 2,
            5, 4, 3,
            5, 1, 4
        };

        // Create a new mesh and populate it with vertices and triangles
        Mesh mesh = new Mesh();
        mesh.vertices = vertices;
        mesh.triangles = triangles;
        mesh.RecalculateNormals(); // This is important for proper lighting

        // Set up game object with mesh
        GameObject diamond = new GameObject("Diamond", typeof(MeshFilter), typeof(MeshRenderer));

        // Assign the mesh to the MeshFilter
        diamond.GetComponent<MeshFilter>().mesh = mesh;

        // Assign material to the MeshRenderer
        diamond.GetComponent<MeshRenderer>().material = diamondMaterial;

        // Optionally add a MeshCollider for interaction
        MeshCollider meshCollider = diamond.AddComponent<MeshCollider>();
        meshCollider.sharedMesh = mesh; // Use the same mesh for the collider
        return mesh;
    }


    public void ExtrudeFace(Mesh mesh, int[] faceVertexIndices, float extrudeDistance)
    {
        // The input `faceVertexIndices` should be the indices of the vertices that form the face you want to extrude

        // Get the existing vertices and triangles
        Vector3[] oldVertices = mesh.vertices;
        int[] oldTriangles = mesh.triangles;

        // Calculate the normal of the face to extrude along
        Vector3 faceNormal = CalculateFaceNormal(oldVertices, faceVertexIndices);

        // Duplicate vertices and move them along the normal
        Vector3[] newVertices = new Vector3[oldVertices.Length + faceVertexIndices.Length];
        oldVertices.CopyTo(newVertices, 0);
        int newVertexStartIndex = oldVertices.Length;
        for (int i = 0; i < faceVertexIndices.Length; i++)
        {
            newVertices[newVertexStartIndex + i] = oldVertices[faceVertexIndices[i]] + faceNormal * extrudeDistance;
        }

        // Create new triangles for the sides of the extrusion
        List<int> newTriangles = new List<int>(oldTriangles);
        for (int i = 0; i < faceVertexIndices.Length; i++)
        {
            // Each side is a quad, made of two triangles
            int nextIndex = (i + 1) % faceVertexIndices.Length;
            newTriangles.Add(faceVertexIndices[i]);
            newTriangles.Add(newVertexStartIndex + i);
            newTriangles.Add(faceVertexIndices[nextIndex]);

            newTriangles.Add(faceVertexIndices[nextIndex]);
            newTriangles.Add(newVertexStartIndex + i);
            newTriangles.Add(newVertexStartIndex + nextIndex);
        }

        // Update the mesh
        mesh.vertices = newVertices;
        mesh.triangles = newTriangles.ToArray();
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();

        // Update the collider (optional, if needed)
        MeshCollider meshCollider = GetComponent<MeshCollider>();
        if (meshCollider)
        {
            meshCollider.sharedMesh = null;
            meshCollider.sharedMesh = mesh;
        }
    }

    private Vector3 CalculateFaceNormal(Vector3[] vertices, int[] faceVertexIndices)
    {
        // Calculate the normal based on the vertices provided
        Vector3 a = vertices[faceVertexIndices[0]];
        Vector3 b = vertices[faceVertexIndices[1]];
        Vector3 c = vertices[faceVertexIndices[2]];

        // Cross product of two sides of the triangle
        Vector3 side1 = b - a;
        Vector3 side2 = c - a;

        // The normal is perpendicular to the cross product
        Vector3 normal = Vector3.Cross(side1, side2).normalized;
        return normal;
    }

    // Update is called once per frame
    void Update()
    {
    }


 
}
