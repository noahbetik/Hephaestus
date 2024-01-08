using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AxisGizmo : MonoBehaviour
{
    public LineRenderer xAxis, yAxis, zAxis;
    public float axisLength = 10000f;

    void Update()
    {
        Vector3 center = transform.position;

        //Extend X Axis in both directions
        xAxis.SetPosition(0, center - transform.right * axisLength);
        xAxis.SetPosition(1, center + transform.right * axisLength);

        //Extend Y Axis in both directions
        yAxis.SetPosition(0, center - transform.up * axisLength);
        yAxis.SetPosition(1, center + transform.up * axisLength);

        //Extend Z Axis in both directions
        zAxis.SetPosition(0, center - transform.forward * axisLength);
        zAxis.SetPosition(1, center + transform.forward * axisLength);
    }
}
