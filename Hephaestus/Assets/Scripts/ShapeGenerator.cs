using System.Collections.Generic;
using UnityEngine;

public class ShapeGenerator : MonoBehaviour
{
    [SerializeField] private Material shapeMaterial; // Assign this in the inspector

    // Start is called before the first frame update
    void Start()
    {
        Vector3 p1 = new Vector3(1.2f, 2.3f, 3.4f);
        Vector3 p2 = new Vector3(4.5f, 5.6f, 6.7f);

        GenerateLine(p1, p2);
        CreateSquarePlane();
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

    // Update is called once per frame
    void Update()
    {
    }


 
}
