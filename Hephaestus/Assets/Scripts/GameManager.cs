using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UIElements;
using Slider = UnityEngine.UI.Slider;


public class GameManager : MonoBehaviour
{
    [SerializeField] public GameObject cube;

    [SerializeField] public Slider scaleSlider;
    [SerializeField] public Slider rotate_x_Slider;
    [SerializeField] public Slider rotate_y_Slider;
    [SerializeField] public Slider extrustion_y_Slider;
    [SerializeField] public Slider extrusion_x_Slider;
    [SerializeField] public Slider extrusion_x_Slider_neg;


    [SerializeField] public float speed = 10f;
    private Vector3 originalScale;
    private Vector3 originalPosition;
    public Vector3 scaleVector;
    private Vector3[] originalVertices;

    private float extrusionFactorX = 1.0f; 
    private float uniformScaleFactor = 1.0f; 
    private float extrusionPositive = 0.0f; 
    private float extrusionNegative = 0.0f; 
    private float xRotation = 0f;
    private float yRotation = 0f;

    public MeshFilter meshFilter;
    void Start()

    {

        if (!meshFilter)
        {
            meshFilter = cube.GetComponent<MeshFilter>();
        }

        originalVertices = meshFilter.mesh.vertices.Clone() as Vector3[];
        originalScale = transform.localScale;
        originalPosition = cube.transform.position;

        // Initialize the sliders and object state
        extrusion_x_Slider.value = 0; 
        scaleSlider.value = 1;


    }

    void Update()
    {

    }

    public void UpdateExtrusion()
    {
        // Update extrusion amounts based on sliders
        extrusionPositive = extrusion_x_Slider.value;
        extrusionNegative = extrusion_x_Slider_neg.value;

        Vector3[] newVertices = new Vector3[originalVertices.Length];
        Array.Copy(originalVertices, newVertices, originalVertices.Length);

        for (int i = 0; i < newVertices.Length; i++)
        {
            if (originalVertices[i].x >= 0)
            {
                // Apply positive extrusion
                newVertices[i].x = originalVertices[i].x + extrusionPositive;
            }
            else
            {
                // Apply negative extrusion
                newVertices[i].x = originalVertices[i].x - extrusionNegative;
            }
        }

        // Update the mesh with the new vertices
        meshFilter.mesh.vertices = newVertices;
        meshFilter.mesh.RecalculateBounds();
        meshFilter.mesh.RecalculateNormals();
    }

    public void UpdateScale()
    {
        // Update the uniform scale factor based on the slider value
        uniformScaleFactor = scaleSlider.value;

        // Call the unified update method
        UpdateScaleAndExtrusion();
    }

    private void UpdateScaleAndExtrusion()
    {
        // Apply both extrusion and uniform scaling
        Vector3 newScale = new Vector3(originalScale.x * extrusionFactorX * uniformScaleFactor,
                                       originalScale.y * uniformScaleFactor,
                                       originalScale.z * uniformScaleFactor);

        // Update the scale of the cube
        cube.transform.localScale = newScale;

        // Adjust the position for extrusion
        float positionOffset = (originalScale.x * extrusionFactorX - originalScale.x) / 2;
        cube.transform.position = new Vector3(originalPosition.x + positionOffset, originalPosition.y, originalPosition.z);
    }
    public void SliderUpdate()
    {
        cube.transform.localScale = (scaleVector);
    }
    private float rotationSpeed = 1500f; 

    public void Rotate_x_Update()
    {
        xRotation = rotate_x_Slider.value; 
        ApplyRotation();
    }

    public void Rotate_y_Update()
    {
        yRotation = rotate_y_Slider.value; 
        ApplyRotation();
    }

    private void ApplyRotation()
    {
        // Create separate Quaternions for X and Y rotations
        Quaternion xRot = Quaternion.Euler(xRotation, 0, 0);
        Quaternion yRot = Quaternion.Euler(0, yRotation, 0);

        // Combine the rotations
        cube.transform.rotation = xRot * yRot;
    }



}