import open3d as o3d
import win32gui
import numpy as np
import socket as s
import time
import sys
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtGui import QFont


from camera_configs import predefined_extrinsics
from camera_configs import forward_vectors
from camera_configs import faces



# ----------------------------------------------------------------------------------------
# SHORTCUTS
# ----------------------------------------------------------------------------------------
#
# Ctrl + K + C --> comment block
#
# ----------------------------------------------------------------------------------------
# GLOBAL VARIABLES
# ----------------------------------------------------------------------------------------

snapCount = 0
objects_dict = {}
curr_highlighted = False
prevRotated = False
prevSnapped = False
marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
previous_look_at_point = None
zoomFactor = 0.25
extrusion_distance = 0
view_axis = ''

# ----------------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------------------------




class Open3DVisualizerWidget(QtWidgets.QWidget):
    def __init__(self, vis, parent=None):
        super(Open3DVisualizerWidget, self).__init__(parent)
        self.vis = vis
        self.initUI()
        
    def initUI(self):
        # Assuming your Open3D window has been created with vis.create_window()
        hwnd = win32gui.FindWindowEx(0, 0, None, "Open3D")  # Find the Open3D window; might need adjustment
        self.window = QtGui.QWindow.fromWinId(hwnd)
        self.windowcontainer = QtWidgets.QWidget.createWindowContainer(self.window)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.windowcontainer)
        self.setLayout(layout)

    def closeEvent(self, event):
        super(Open3DVisualizerWidget, self).closeEvent(event)
        self.vis.destroy_window()

class TextDisplayWidget(QtWidgets.QLabel):
    def __init__(self, text="Hello World", parent=None):
        super(TextDisplayWidget, self).__init__(parent)
               #
        font = QFont("Arial", 24)  
        self.setFont(font)

        self.setText(text)
        self.setAlignment(QtCore.Qt.AlignCenter)  # Center align the text

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, vis):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Open3D and Text Display in PySide6")
        self.setGeometry(100, 100, 1280, 960)
        self.setStyleSheet("background-color: darkgray;") 


        # Create a central widget and set a layout for it
        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)

        # Create and add the text display widget
        self.text_display_widget = TextDisplayWidget("Hello World")
        layout.addWidget(self.text_display_widget)  # No stretch factor necessary here

        # Create and add the Open3D visualizer widget
        self.open3d_widget = Open3DVisualizerWidget(vis)
        layout.addWidget(self.open3d_widget, 1)  # Add with stretch factor of 1 to take up remaining space


    def update_text(self, new_text):
        self.text_display_widget.setText(new_text)


def create_grid(size=10, n=10, plane='xz', color=[0.5, 0.5, 0.5]):
    """
    Create a grid in the specified plane.
    
    Args:
    - size (float): The length of the grid.
    - n (int): The number of divisions in the grid.
    - plane (str): The plane on which the grid is aligned ('xy', 'xz', or 'yz').
    - color (list): The color of the grid lines.
    
    Returns:
    - o3d.geometry.LineSet: The grid as a line set.
    """
    lines = []
    points = []
    line_color = []

    # Determine points and lines based on the specified plane
    if plane == 'xz':
        for i in range(n+1):
            points.append([i * size / n - size / 2, 0, -size / 2])
            points.append([i * size / n - size / 2, 0, size / 2])
            lines.append([2*i, 2*i+1])
            
            points.append([-size / 2, 0, i * size / n - size / 2])
            points.append([size / 2, 0, i * size / n - size / 2])
            lines.append([2*(n+1)+2*i, 2*(n+1)+2*i+1])
    
    # Repeat the above logic for 'xy' and 'yz' planes if necessary

    for _ in range(len(lines)):
        line_color.append(color)

    # Create a LineSet object from the points and lines
    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(points)
    grid.lines = o3d.utility.Vector2iVector(lines)
    grid.colors = o3d.utility.Vector3dVector(line_color)
    
    return grid


def closest_config(current_extrinsic, extrinsics=predefined_extrinsics):
    
    current_rotation = current_extrinsic[:3, :3]
     #find closest match
    closest_match = None
    smallest_difference = np.inf

    for name, rotation in extrinsics.items():
        difference = np.sum(np.abs(current_rotation - rotation))
        if difference < smallest_difference:
            closest_match = name
            smallest_difference = difference
    
    print(closest_match)
    return closest_match

def identify_plane(current_extrinsic):
    extr = np.array(current_extrinsic) # is this already a np.array?
    closest_plane = closest_config(extr)

    # Major axes directions
    major_axes = {
        'x': np.array([1, 0, 0]),
        '-x': np.array([-1, 0, 0]),
        'y': np.array([0, 1, 0]),
        '-y': np.array([0, -1, 0]),
        'z': np.array([0, 0, 1]),
        '-z': np.array([0, 0, -1])
    }

    # Find the major axis that the camera is pointing along
    for axis, direction in major_axes.items():
        if np.allclose(np.dot(extr, [0, 0, -1]), direction):
            print(f"Camera is looking along the {axis}-axis")
            return axis


    

def snap_to_closest_plane(vis, view_control):
    global prevSnapped
    
    # if (prevSnapped): 
    #     snap_isometric(vis, view_control) 
    #     return
    cam_params = view_control.convert_to_pinhole_camera_parameters()
    current_extrinsic = cam_params.extrinsic

   
    print("************current extrisnic: ", current_extrinsic)

    closest_match = closest_config(current_extrinsic)
    
    updated_extrinsic = current_extrinsic.copy()
    updated_extrinsic[:3, :3] = predefined_extrinsics[closest_match]

    smooth_transition(vis, view_control, updated_extrinsic)
    prevSnapped = True


def snap_isometric(vis, view_control):
    global prevSnapped
    # Obtain the current extrinsic parameters
    cam_params = view_control.convert_to_pinhole_camera_parameters()
    current_extrinsic = cam_params.extrinsic
    #print("Current Extrinsic:", current_extrinsic)

    #predefined extrinsic set for the isometric view
    target_extrinsic = np.array([
        [0.86600324, 0.0, 0.50003839, -0.1],
        [-0.21133220, -0.90629373, 0.36601965, 0.1],
        [0.45318549, -0.42264841, -0.78485109, 0.75],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    # Execute the smooth transition to the new isometric view
    smooth_transition(vis, view_control, target_extrinsic)
    prevSnapped = False


def convert_rotation_matrix_to_quaternion(rotation_matrix):
    # Ensure the rotation matrix is writable by making a copy
    rotation_matrix_copy = np.array(rotation_matrix, copy=True)
    return R.from_matrix(rotation_matrix_copy).as_quat()


def convert_quaternion_to_rotation_matrix(quaternion):
    return R.from_quat(quaternion).as_matrix()


def quaternion_slerp(quat_start, quat_end, fraction):
    # Create a times series for keyframe quaternions
    key_times = [0, 1]  # Start and end times
    key_rots = R.from_quat([quat_start, quat_end])  # Keyframe rotations

    # Create the interpolator object
    slerp = Slerp(key_times, key_rots)

    # Interpolate the rotation at the given fraction
    interp_rot = slerp([fraction])

    # Return the interpolated quaternion
    return interp_rot.as_quat()[0]  # Slerp returns an array of rotations


def smooth_transition(vis, view_control, target_extrinsic, steps=100):
    current_extrinsic = view_control.convert_to_pinhole_camera_parameters().extrinsic
    current_translation = current_extrinsic[:3, 3]
    target_translation = target_extrinsic[:3, 3]

    # Convert rotation matrices to quaternions
    current_quat = convert_rotation_matrix_to_quaternion(current_extrinsic[:3, :3])
    target_quat = convert_rotation_matrix_to_quaternion(target_extrinsic[:3, :3])

    for step in range(steps):
        fraction = step / float(steps)

        # Interpolate translation linearly
        interp_translation = (
            current_translation + (target_translation - current_translation) * fraction
        )

        # Interpolate rotation using slerp
        interp_quat = quaternion_slerp(current_quat, target_quat, fraction)
        interp_rotation_matrix = convert_quaternion_to_rotation_matrix(interp_quat)

        # Construct the interpolated extrinsic matrix
        interp_extrinsic = np.eye(4)
        interp_extrinsic[:3, :3] = interp_rotation_matrix
        interp_extrinsic[:3, 3] = interp_translation

        # Set the new extrinsic matrix
        cam_params = view_control.convert_to_pinhole_camera_parameters()
        cam_params.extrinsic = interp_extrinsic
        view_control.convert_from_pinhole_camera_parameters(cam_params, True)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.0001)


def move_camera(view_control, direction, amount=1.5):
    cam_params = view_control.convert_to_pinhole_camera_parameters()
    extrinsic = np.array(cam_params.extrinsic)

    if direction == "forward":
        extrinsic[2, 3] -= amount
    elif direction == "backward":
        extrinsic[2, 3] += amount
    elif direction == "left":
        extrinsic[0, 3] -= amount
    elif direction == "right":
        extrinsic[0, 3] += amount
    elif direction == "up":
        extrinsic[1, 3] -= amount
    elif direction == "down":
        extrinsic[1, 3] += amount

    print(extrinsic[0, 3])

    cam_params.extrinsic = extrinsic
    view_control.convert_from_pinhole_camera_parameters(cam_params, True)


def move_camera_v2(view_control, direction, amount=1.5):
    global zoomFactor
    cam_params = view_control.convert_to_pinhole_camera_parameters()
    extrinsic = np.array(cam_params.extrinsic)

    if direction == "x":
        extrinsic[2, 3] += amount
    elif direction == "y":
        extrinsic[0, 3] += amount
    elif direction == "z":
        extrinsic[1, 3] += amount

    cam_params.extrinsic = extrinsic
    view_control.convert_from_pinhole_camera_parameters(cam_params, True)
    zoomFactor += amount/2

def move_camera_v3(view_control, vals):
    cam_params = view_control.convert_to_pinhole_camera_parameters()
    extrinsic = np.array(cam_params.extrinsic)

    # x axis handled by zoom
    extrinsic[0, 3] += vals[0] # y axis
    extrinsic[1, 3] += vals[1] # z axis

    cam_params.extrinsic = extrinsic
    view_control.convert_from_pinhole_camera_parameters(cam_params, True)


def rotate_camera(view_control, axis, degrees=5):
    global zoomFactor
    
    
    
  
    #else:
        #zoomFactor *= (1 + 0.05)  # Slightly increase for "zooming in"

    cam_params = view_control.convert_to_pinhole_camera_parameters()
    R = cam_params.extrinsic[:3, :3]
    t = cam_params.extrinsic[:3, 3].copy()
    angle = np.radians(degrees)

    if axis == "y":
        
        if degrees > 0:  # Assuming positive degrees tilt the view upwards
              zoomFactor *= (1)  # Slightly decrease for "zooming out"
        rotation_matrix = np.array(
            [       
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)],
            ]
        )
    elif axis == "x":
        rotation_matrix = np.array(
            [
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)],
            ]
        )
    elif axis == "z":
        rotation_matrix = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )


    R = rotation_matrix @ R
    new_extrinsic = np.eye(4)
    new_extrinsic[:3, :3] = R
    new_extrinsic[:3, 3] = t

    cam_params.extrinsic = new_extrinsic  # Set the new extrinsic matrix
    view_control.convert_from_pinhole_camera_parameters(cam_params, True)


def axis2arr(axis, delta, alpha):
    match axis:
        case "x":
            return [delta * alpha, 0, 0]
        case "y":
            return [0, delta * alpha, 0]
        case "z":
            return [0, 0, delta * alpha]


def smartConnect(endPoint, startPoint):
    threshold = 0.1  # tune
    if (
        abs(endPoint[0] - startPoint[0]) < threshold
        and abs(endPoint[1] - startPoint[1]) < threshold
    ):
        return startPoint
    else:
        return endPoint

def smartConnectBool(endPoint, startPoint):
    threshold = 0.1  # tune
    if (
        abs(endPoint[0] - startPoint[0]) < threshold
        and abs(endPoint[1] - startPoint[1]) < threshold
    ):
        return True
    else:
        return False


def startClient():
    # should have error control
    clientSocket = s.socket(s.AF_INET, s.SOCK_STREAM)
    serverAddr = ("localhost", 444)  # (IP addr, port)
    clientSocket.connect(serverAddr)

    return clientSocket

def startServer():
    serverSocket = s.socket(s.AF_INET, s.SOCK_STREAM) # IPV4 address family, datastream connection
    serverAddr = ('localhost', 4445) # (IP addr, port)
    serverSocket.bind(serverAddr)
    print("Server started on " + str(serverAddr[0]) + " on port " + str(serverAddr[1]))

    return serverSocket

def makeConnection(serverSocket):
    serverSocket.listen(1) # max number of queued connections
    print("Waiting for connection...")
    clientSocket, clientAddr = serverSocket.accept()
    print("\nConnection from " + str(clientAddr[0]) + " on port " + str(clientAddr[1]))
    
    return clientSocket

tcp_command_buffer = ""

def getTCPData(sock):
    global tcp_command_buffer
    try:
        # Temporarily set a non-blocking mode to check for new data
        sock.setblocking(False)
        data = sock.recv(1024)  # Attempt to read more data
        
        if data:
        # Process received data
            print(f"Received: {data.decode('ascii').strip()}")

            # Send acknowledgment back
            sock.sendall("ACK".encode("ascii"))
        #sock.setblocking(True)  # Revert to the original blocking mode

        # Decode and append to buffer
        tcp_command_buffer += data.decode('ascii')

        # Process if delimiter is found
        if '\n' in tcp_command_buffer:
            command, tcp_command_buffer = tcp_command_buffer.split('\n', 1)
            command = command.strip("$")  # Strip any '$' characters as in your original processing
            print("Received: ", command)
            return command
    except BlockingIOError:
        # No new data available; pass this iteration
        pass
    except s.timeout:
        # Handle possible timeout exception if setblocking(false) wasn't enough
        pass
    except Exception as e:
        print(f"Unexpected error receiving data: {e}")

    # No complete command was processed
    return None


def closeClient(sock):
    sock.close()

def vectorDistance(p1, p2):
    # Extract coordinates
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    
    # Calculate the distance
    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) ** 0.5
    return distance

# ----------------------------------------------------------------------------------------
# PARSE COMMAND
# ----------------------------------------------------------------------------------------


def parseCommand(
    command, view_control, camera_parameters, vis, geometry_dir, history, objects_dict, ls_dict, counters, main_window
):
    # FORMAT
    # [command] [subcommand]

    global prevRotated
    global snapCount
    info = command.split(" ")
    print(info)
    objectHandle = ""    
    object_id = ""

    # Check for selected objects in objects_dict and update objectHandle to point to the mesh if selected
    for id, obj_info in objects_dict.items():
        if object_id == 'original':  # Skip the 'original' entry
            continue
        if obj_info.get('selected', False):  # Check if the 'selected' key exists and is True
            objectHandle = obj_info['object']  # Now, objectHandle directly references the mesh object
            object_id = id
            print(f"Selected object: {id}")
            break  # Assume only one object can be selected at a time; break after finding the first selected object


  

    match info[0]:
        case "motion":
            snapCount=0

            if objectHandle == "":
                handleCam(info[1:], view_control, history, vis)
                return ""
            else:
                # if (prevRotated) : snap_to_closest_plane(vis, view_control)
                # prevRotated = False
                handleUpdateGeo(info[1:], history, objectHandle, vis, main_window, objects_dict, object_id)
                return ""
        case "select":
            snapCount = 0

            return handleSelection(objects_dict, vis, main_window)  # Assume this function handles object selection
        case "deselect":
            snapCount = 0

            return handleDeselection(objects_dict, vis, main_window)  # Assume this function handles object selection
        case "create":
            snapCount = 0

            handleNewGeo(info[1:], view_control, camera_parameters, vis, objects_dict, ls_dict, counters)
            return ""
        case "update":
            snapCount = 0

            # if objectHandle:
            #     if (prevRotated) : snap_to_closest_plane(vis, view_control)
            #     prevRotated = False
            handleUpdateGeo(info[1:], history, objectHandle, vis, main_window, objects_dict, object_id)
            
        case "home":
            snap_isometric(vis, view_control)
        case "snap":
            snapCount+=1
            if (snapCount>2):
                snap_isometric(vis, view_control)
            else: snap_to_closest_plane(vis, view_control)
        case "delete":
        
        #for extrusion, reset to original object
            print("last operation was ", history["operation"])
            print("reverting to original state")
            if 'original' in objects_dict[object_id]:
                history['total_extrusion_x'] = 0
                history['total_extrusion_y'] = 0

                vis.remove_geometry(objectHandle, reset_bounding_box=False)
                objectHandle = clone_mesh(objects_dict[object_id]['original'])
                objectHandle.compute_vertex_normals()
                objectHandle.paint_uniform_color([0.540, 0.68, 0.52])  #back to green 
                vis.add_geometry(objectHandle, reset_bounding_box=False)
                vis.update_geometry(objectHandle)


                objects_dict[object_id]['object'] = objectHandle  # Update the current object with the original
                history['last_extrusion_distance_x'] = 0.0
                history['last_extrusion_distance_y'] = 0.0

                objects_dict[object_id]['total_extrusion_x'] = 0.0
                objects_dict[object_id]['total_extrusion_y'] = 0.0
            
            





def highlight_objects_near_camera(vis, view_control, objects_dict):
    # Get the camera position from the view control
    cam_params = view_control.convert_to_pinhole_camera_parameters()
    closest_match = closest_config(cam_params.extrinsic)
    extrinsic = predefined_extrinsics[closest_match]

    
    forward_vector = forward_vectors[closest_match]
    print("forward vector is ",forward_vector)

    
    camera_position = np.array(cam_params.extrinsic[:3, 3]) 
    print("*************original camera position is at", camera_position)
    camera_position[0] *= forward_vector[0]
    camera_position[1] *= forward_vector[1]
    camera_position[2] *= forward_vector[2]




    # Initialize the closest distance and the object ID
    closest_distance = np.inf
    closest_object_id = None

    # Find the object closest to the camera
    for object_id, info in objects_dict.items():
        if object_id == 'original':  # Skip the 'original' entry
            continue
        obj = info['object']
        centroid = info['center']

        distance = np.linalg.norm(camera_position[:2] - centroid[:2])       
        
        print("*************camera position is at", camera_position)

        print("**************************object centred at ",centroid)
        
        if distance < closest_distance:
            closest_distance = distance
            closest_object_id = object_id

    # Highlight the closest object and unhighlight others
    for object_id, info in objects_dict.items():
        if object_id == 'original':  # Skip the 'original' entry
            continue
        obj = info['object']
        if object_id == closest_object_id:
            # Highlight the closest object
            obj.paint_uniform_color([0.640, 0.91, 0.62])  # Light green for highlighted object
            info['highlighted'] = True
        else:
            # Unhighlight all other objects by resetting their color
            obj.paint_uniform_color([0.5, 0.5, 0.5])  # Default color
            info['highlighted'] = False
        
        # Update the geometry to reflect changes
        vis.update_geometry(obj)

    # You might need to call this if the visualizer doesn't automatically refresh
    vis.update_renderer()


def removeGeometry(vis, object):
    vis.remove_geometry(object)
    return

def addGeometry(vis, obj):
    obj.paint_uniform_color([0.5, 0.5, 0.5])  # Reset the object color to grey
    vis.add_geometry(obj)
    return
              


def scale_object(objectHandle, delta):
    scaleFactor = 1 + delta
    
    # Compute the object's center for uniform scaling about its center
    print("increasing object scale by factor of ", delta)
    center = objectHandle.get_center()

    objectHandle.scale(scaleFactor, center)
    # Apply the transformation


def rotate_object(objectHandle, axis, degrees=5):
    # Calculate the rotation angle in radians
    angle = np.radians(degrees)
    
    # Compute the object's center
    center = objectHandle.get_center()
    
    # Define the rotation matrix for each axis
    if axis == "x":
        print("rotating object about x")
        rotation_matrix = np.array([
            [1, 0, 0, 0],
            [0, np.cos(angle), -np.sin(angle), 0],
            [0, np.sin(angle), np.cos(angle), 0],
            [0, 0, 0, 1]
        ])
    elif axis == "y":
        print("rotating object about y")

        rotation_matrix = np.array([
            [np.cos(angle), 0, np.sin(angle), 0],
            [0, 1, 0, 0],
            [-np.sin(angle), 0, np.cos(angle), 0],
            [0, 0, 0, 1]
        ])
    else:
        raise ValueError("Axis must be 'x' or 'y'")
    
    # Translate the object to the origin, rotate, and then translate back
    translate_to_origin = np.eye(4)
    translate_to_origin[:3, 3] = -center
    translate_back = np.eye(4)
    translate_back[:3, 3] = center
    
    # Combine the transformations
    transformation = translate_back @ rotation_matrix @ translate_to_origin
    
    # Apply the transformation
    objectHandle.transform(transformation)
    
    
def handleCam(subcommand, view_control, history, vis):
    
    global prevRotated
    global marker

    # FORMAT:
    # start [operation] [axis] [position]
    # position n
    # position n+1
    # position n+2
    # ...
    # position n+m
    # end [operation] [axis]

    # history: {operation:operation, axis:axis, lastVal:lastVal}

    alphaM = 0.01  # translation scaling factor
    alphaR = 1  # rotation scaling factor
    alphaZ = 0.01  # zoom scaling factor

    match subcommand[1]:
        case "start":
            print("starting motion")
            history["operation"] = subcommand[0] # in theory don't need to store this anymore since optype sent each update
            if history["operation"] != "rotate":
                history["lastVal"] = subcommand[2]
            else:
                history["axis"] = subcommand[2]
                history["lastVal"] = subcommand[3]
            print(history)
        case "end":
            print("ending motion")
            history["operation"] = ""
            history["axis"] = ""
            history["lastVal"] = ""
            return
        case "position":
            match history["operation"]:
                case "pan":
                    print("camera pan update")
                    splitCoords = subcommand[2].strip("()").split(",")
                    oldCoords = history["lastVal"].strip("()").split(",")
                    deltaY = (float(splitCoords[0]) - float(oldCoords[0])) * alphaM
                    deltaZ = (float(splitCoords[1]) - float(oldCoords[1])) * alphaM
                    move_camera_v3(view_control, [deltaY, deltaZ])

                    highlight_objects_near_camera(vis, view_control, objects_dict)
                    
                case "rotate":
                    print("current axis, ", history["axis"])
                    prevRotated = True
                    print("camera rotate update")
                    delta = float(subcommand[2]) - float(history["lastVal"])
                    rotate_camera(view_control, history["axis"], degrees=delta * alphaR)
                    highlight_objects_near_camera(vis, view_control, objects_dict)


                case "zoom":
                    print("camera zoom update")
                    delta = float(subcommand[2]) - float(history["lastVal"])
                    move_camera_v2(view_control, "x", delta * alphaZ)
                    highlight_objects_near_camera(vis, view_control, objects_dict)

            history["lastVal"] = subcommand[2]
        case _:
            print("INVALID COMMAND")


def handleNewGeo(subcommand, view_control, camera_parameters, vis, geometry_dir, ls_dict, counters):
    
    global view_axis

    alphaL = 0.002 # line scaling factor (maybe better way to do this)
    #print("***********subcommand[0] ", subcommand[0])
    #print("***********subcommand[1] ", subcommand[1])
    


    match subcommand[0]:
        case "box":
            print("Creating new box with dimensions ___ at ___")
            coords = list(map(float, subcommand[1].strip("()").split(",")))
            dimensions = list(map(float, subcommand[2].strip("()").split(",")))

            # Store the current view matrix
            current_view_matrix = (
                view_control.convert_to_pinhole_camera_parameters().extrinsic
            )

            new_box = o3d.geometry.TriangleMesh.create_box(
                width=dimensions[0], height=dimensions[1], depth=dimensions[2]
            )
            new_box.compute_vertex_normals()
            new_box.translate(np.array(coords))
            addGeometry(vis,new_box)
            name = "mesh" + str(geometry_dir["counters"]["mesh"])
            geometry_dir[name] = new_box
            geometry_dir["counters"]["mesh"] += 1

            # Set back the stored view matrix
            camera_parameters.extrinsic = current_view_matrix
            view_control.convert_from_pinhole_camera_parameters(camera_parameters, True)
        case "line":  # line handling not fully implemented yet
            # Store the current view matrix
            current_view_matrix = (
                view_control.convert_to_pinhole_camera_parameters().extrinsic
            )
            #print("**********SUBCOMMAND IS ",subcommand[1])

            if subcommand[1] == "start":

                closest_plane = predefined_extrinsics[closest_config(current_view_matrix)]
                view_axis = identify_plane(closest_plane) # global var
                
                pcd = o3d.geometry.PointCloud()
                ls = o3d.geometry.LineSet()

                print("Creating new line with endpoints ___ and ___")

                match view_axis: # not sure if -ve and +ve can be treated as same
                    case 'x':
                        print("x")
                        coords1 = [0.0] + [float(val)*alphaL for val in subcommand[2].strip("()").split(",")]
                        coords2 = [0.0] + [float(val)*alphaL for val in subcommand[3].strip("()").split(",")]
                    case '-x':
                        print("-x")
                        coords1 = [0.0] + [float(val)*alphaL for val in subcommand[2].strip("()").split(",")]
                        coords2 = [0.0] + [float(val)*alphaL for val in subcommand[3].strip("()").split(",")]
                    case 'y':
                        print("y")
                        coords1 = [float(val)*alphaL for val in subcommand[2].strip("()").split(",")].insert(1, [0.0])
                        coords2 = [float(val)*alphaL for val in subcommand[3].strip("()").split(",")].insert(1, [0.0])
                    case '-y':
                        print("-y")
                        coords1 = [float(val)*alphaL for val in subcommand[2].strip("()").split(",")].insert(1, [0.0])
                        coords2 = [float(val)*alphaL for val in subcommand[3].strip("()").split(",")].insert(1, [0.0])
                    case 'z':
                        print("z")
                        coords1 = [float(val)*alphaL for val in subcommand[2].strip("()").split(",")] + [0.0]
                        coords2 = [float(val)*alphaL for val in subcommand[3].strip("()").split(",")] + [0.0]
                    case '-z':
                        print("-z")
                        coords1 = [float(val)*alphaL for val in subcommand[2].strip("()").split(",")] + [0.0]
                        coords2 = [float(val)*alphaL for val in subcommand[3].strip("()").split(",")] + [0.0]

                points = np.array([coords1, coords2]) 
                lines = np.array([[0, 1]])

                # Store the current view matrix
                current_view_matrix = (
                    view_control.convert_to_pinhole_camera_parameters().extrinsic
                )

                pcd.points = o3d.utility.Vector3dVector(points)

                ls.points = o3d.utility.Vector3dVector(points)
                ls.lines = o3d.utility.Vector2iVector(lines)

                vis.add_geometry(pcd)
                vis.add_geometry(ls)

                lsName = "ls" + str(counters["ls"])
                counters["ls"] += 1

                pcdName = "pcd" + str(counters["pcd"])
                counters["pcd"] += 1

                ls_dict[lsName] = ls
                ls_dict[pcdName] = pcd


                # Set back the stored view matrix
                camera_parameters.extrinsic = current_view_matrix
                view_control.convert_from_pinhole_camera_parameters(
                    camera_parameters, True
                )

            elif subcommand[1] == "end":
        
                ls_id = "ls" + str(counters["ls"] - 1)
                pcd_id = "pcd" + str(counters["pcd"] - 1)
                ls = ls_dict[ls_id]
                pcd = ls_dict[pcd_id]
                print("Ending line")
                # threshold for connecting closed-loop geometry
                all_points = np.asarray(pcd.points).tolist()
                print(pcd.points)
                print(all_points)
                if smartConnectBool(all_points[-1], all_points[0]):
                    print("CONNECTY WECTY")
                    ls.lines.extend(o3d.utility.Vector2iVector(np.array([[len(pcd.points), 0]])))
                else:
                    print("NO CONNECTY WECTY")

                vis.update_geometry(pcd)
                vis.update_geometry(ls)

                view_axis = ''
                

            else:
                print("still sketching")
                ls_id = "ls" + str(counters["ls"] - 1)
                pcd_id = "pcd" + str(counters["pcd"] - 1)

                ls = ls_dict[ls_id]
                pcd = ls_dict[pcd_id]
                
                print(view_axis)


                match view_axis: # not sure if -ve and +ve can be treated as same
                    case 'x':
                        print("x")
                        new_points = [0.0] + [float(val)*alphaL for val in subcommand[1].strip("()").split(",")]
                    case '-x':
                        print("-x")
                        new_points = [0.0] + [float(val)*alphaL for val in subcommand[1].strip("()").split(",")]
                    case 'y':
                        print("y")
                        new_points = [float(val)*alphaL for val in subcommand[1].strip("()").split(",")].insert(1, [0.0])
                    case '-y':
                        print("-y")
                        new_points = [float(val)*alphaL for val in subcommand[1].strip("()").split(",")].insert(1, [0.0])
                    case 'z':
                        print("z")
                        new_points = [float(val)*alphaL for val in subcommand[1].strip("()").split(",")] + [0.0]
                    case '-z':
                        print("-z")
                        new_points = [float(val)*alphaL for val in subcommand[1].strip("()").split(",")] + [0.0]
                        
                prev_points = np.asarray(ls.points).tolist() # tune number of prev elements to consider

                vd1 = vectorDistance(prev_points[-2], new_points)
                vd2 = vectorDistance(prev_points[-1], prev_points[-2])
                
                if (vd1 != 0) and (vd1 < 10*vd2 or vd2 == 0): # or (vd1 > 10*vd2 and vd2 != 0): #(vd1 > 10*vd2 or vd2 == 0):
                    add_this = o3d.utility.Vector3dVector(
                        np.array([new_points])
                    )
                    pcd.points.extend(add_this)  # get proper references for this
                    ls.points.extend(add_this)  # get proper references for this
                    ls.lines.extend(o3d.utility.Vector2iVector(np.array([[len(pcd.points) - 1, len(pcd.points)]])))

                    vis.update_geometry(pcd)
                    vis.update_geometry(ls)
                else:
                    print(f"vector distances: {vd1}, {vd2}")
                    #return

                


def clone_mesh(mesh):
    cloned_mesh = o3d.geometry.TriangleMesh()
    cloned_mesh.vertices = o3d.utility.Vector3dVector(np.array(mesh.vertices))
    cloned_mesh.triangles = o3d.utility.Vector3iVector(np.array(mesh.triangles))
    cloned_mesh.vertex_normals = o3d.utility.Vector3dVector(np.array(mesh.vertex_normals))
    cloned_mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(mesh.vertex_colors))
    # If you have other attributes like texture coordinates, you'll need to copy them as well
    return cloned_mesh

def handleUpdateGeo(subcommand, history, objectHandle, vis, main_window, objects_dict, object_id):
    alphaM = 0.01  # Translation scaling factor
    alphaR = 1  # Rotation scaling factor (in radians for Open3D)
    alphaS = 100  # Scaling scaling factor
    alphaE = 100 #extrusion scaling factor
    
    
    global extrusion_distance
    global direction

    match subcommand[1]:
        case "start":
            
            history["operation"] = subcommand[0]  # Operation type
            history["axis"] = subcommand[2] if len(subcommand) > 2 else None  # Axis, if applicable
            history["lastVal"] = subcommand[3] if len(subcommand) > 3 else None  # Starting value, if applicable
            
            if (subcommand[0] == "extrude"):
                snap_to_closest_plane(vis, vis.get_view_control())
        case "end":
            #print("Ending motion")
            # Reset the history after operation ends
            
            if(history["operation"] == "extrude"):   
                objectHandle.paint_uniform_color([0.540, 0.68, 0.52])  #back to green 
                vis.update_geometry(objectHandle)            
                history['total_extrusion_x'] = 0
                history['total_extrusion_y'] = 0

                if 'original' in objects_dict:  
                  del objects_dict['original']

        case "position":
                match history["operation"]:
                    case "pan": #not actually pan, but object translation
                        #print("Updating position or transformation")
                        main_window.update_text("translating object")

                        try:
                            coords = subcommand[2].strip("()").split(",")
                            currentX = float(coords[0])
                            currentY = float(coords[1])
                            
                            oldCoords = history["lastVal"].strip("()").split(",")
                            
                            oldX = float(oldCoords[0])
                            oldY = float(oldCoords[1])
                            
                            deltaX = (currentX - oldX) * alphaM
                            deltaY = (currentY - oldY) * alphaM
                            view_control = vis.get_view_control()
                            cam_params = view_control.convert_to_pinhole_camera_parameters()
                            rotation_matrix = np.linalg.inv(cam_params.extrinsic[:3, :3])

                            view_space_translation = np.array([deltaX, deltaY, 0])
                            world_space_translation = rotation_matrix.dot(view_space_translation)
                            
                            objectHandle.translate(world_space_translation, relative=True)
                            objects_dict[object_id]['center'] = objectHandle.get_center()
                            objects_dict[object_id]['original'].translate(world_space_translation, relative=True)

                            #print("Translating object by", world_space_translation)
                            
                            history["lastX"] = currentX
                            history["lastY"] = currentY
                        except Exception as e:
                            #print(f"Error processing numerical values from command: {e}")
                            pass
                           
                    case "rotate": 
                        
                        try:
                            #print("object rotation")
                            delta = float(subcommand[2]) - float(history["lastVal"])
                            rotate_object(objectHandle, history["axis"], degrees=delta * alphaR)
                            rotate_object( objects_dict[object_id]['original'], history["axis"], degrees=delta * alphaR)
                           

                            #print("current degrees ", delta * alphaR)
                        except Exception as e:
                            #print(f"Error processing numerical values from command: {e}")
                            pass

                    case "zoom": 
                        try:
                            #print("object scaling")
                            delta = float(subcommand[2]) - float(history["lastVal"])

                            selected_object_id = None
                            for object_id, obj_info in objects_dict.items():
                                if object_id == 'original':  # Skip the 'original' entry
                                      continue
                                if obj_info.get('selected', False):  
                                    selected_object_id = object_id
                                    obj_info['scale'] += 100 * (delta / alphaS)
                                    break  

                            if selected_object_id is not None:
                                objectScale = objects_dict[selected_object_id]['scale']
                                formatted_text = f"Scaling factor: {objectScale:.0f}%"
                                main_window.update_text(formatted_text)
                            else:
                                #print("No object is currently selected.")
                                pass

                            scale_object(objectHandle, delta/alphaS)       
                            scale_object(objects_dict[object_id]['original'], delta/alphaS)                

                        except Exception as e:
                            #print(f"Error processing numerical values from command: {e}")
                            pass
                                
                    case "extrude":
                        try:
                            
                            coords = subcommand[2].strip("()").split(",")
                            currentX = float(coords[0])
                            currentY = float(coords[1])
                            
                            oldCoords = history["lastVal"].strip("()").split(",")
                            
                            oldX = float(oldCoords[0])
                            oldY = float(oldCoords[1])
                            
                            deltaX = (currentX - oldX) / alphaE
                            deltaY = (currentY - oldY) / alphaE

                            objectHandle.paint_uniform_color([0.53, 0.81, 0.92])  # Set to light blue
                            
                            print("delta x is ",deltaX," delta y is ", deltaY)
                            
                            
                            extrusion_distance_x = deltaX
                            extrusion_distance_y = deltaY
                            history['last_extrusion_distance_x'] += deltaX
                            history['last_extrusion_distance_y'] -= deltaY


                            if 'total_extrusion_x' not in objects_dict[object_id]:
                                objects_dict[object_id]['total_extrusion_x'] = 0.0
                                
                            if 'total_extrusion_y' not in objects_dict[object_id]:
                                objects_dict[object_id]['total_extrusion_y'] = 0.0

                            # Calculate the new total extrusion considering this operation
                            new_total_extrusion_x = objects_dict[object_id]['total_extrusion_x'] + abs(extrusion_distance_x)
                            new_total_extrusion_y = objects_dict[object_id]['total_extrusion_y'] + abs(extrusion_distance_y)
                            print("last extrusion x ",history['last_extrusion_distance_x'])
                            print("last extrusion y ",history['last_extrusion_distance_y'])


                                  
                            # if history['last_extrusion_distance_x'] <= -0.4 or history['last_extrusion_distance_y'] <= -0.4: 
                            #     # Revert to the original state
                            #     print("reverting to original state")
                            #     if 'original' in objects_dict[object_id]:
                            #         history['total_extrusion_x'] = 0
                            #         history['total_extrusion_y'] = 0

                            #         vis.remove_geometry(objectHandle, reset_bounding_box=False)
                            #         objectHandle = clone_mesh(objects_dict[object_id]['original'])
                            #         objectHandle.compute_vertex_normals()
                            #         vis.add_geometry(objectHandle, reset_bounding_box=False)

                            #         objects_dict[object_id]['object'] = objectHandle  # Update the current object with the original
                            #         history['last_extrusion_distance_x'] = 0.0
                            #         history['last_extrusion_distance_y'] = 0.0

                            #         objects_dict[object_id]['total_extrusion_x'] = 0.0
                            #         objects_dict[object_id]['total_extrusion_y'] = 0.0
                            
                            if new_total_extrusion_x > 0.75:
                                print("Maximum extrusion limit in x direction reached. No further extrusion will be performed.")
                                main_window.update_text("Maximum extrusion limit reached. No further extrusion will be performed.")
                                pass

                            elif history['last_extrusion_distance_x'] >= 0.25:##
                                direction = [0,0,1]
                                objects_dict[object_id]['total_extrusion_x'] += 0.2
                                extrude(object_id, objectHandle, objects_dict, vis, history, direction)

                            if new_total_extrusion_y > 0.75:
                                print("Maximum extrusion limit in y direction reached. No further extrusion will be performed.")
                                main_window.update_text("Maximum extrusion limit reached. No further extrusion will be performed.")
                                pass


                            elif history['last_extrusion_distance_y'] >= 0.25:##
                                objects_dict[object_id]['total_extrusion_y'] += 0.2
                                direction = [0,1,0]
                                extrude(object_id, objectHandle, objects_dict, vis, history, direction)

                                #delta = deltaY
                            print("extrusion direction set to ", direction)
                                        
                                




                        except Exception as e:
                            print(f"Error processing numerical values from command: {e}")
                         
                         
                vis.update_geometry(objectHandle)
                history["lastVal"] = subcommand[2]

        case _:
            #print("Invalid command")
            pass



def extrude(object_id, objectHandle, objects_dict, vis, history, direction):
   
        vis.remove_geometry(objectHandle, reset_bounding_box=False)

        # Check and save the original state if not saved yet
        if 'original' not in objects_dict[object_id]:
            objects_dict[object_id]['original'] = clone_mesh(objectHandle)

        direction_tensor = o3d.core.Tensor(direction, dtype=o3d.core.Dtype.Float32)

        # Perform the extrusion
        mesh_extrusion = o3d.t.geometry.TriangleMesh.from_legacy(objectHandle)
        extruded_shape = mesh_extrusion.extrude_linear(direction_tensor, scale=0.2)
        filled = extruded_shape#.fill_holes()

        objectHandle = o3d.geometry.TriangleMesh(filled.to_legacy())
        objectHandle.compute_vertex_normals()
        objectHandle.paint_uniform_color([0.53, 0.81, 0.92])  # Set to light blue
        vis.add_geometry(objectHandle, reset_bounding_box=False)

        objects_dict[object_id]['object'] = objectHandle  # Update the current object
        history['last_extrusion_distance_x'] = 0.0
        history['last_extrusion_distance_y'] = 0.0



def handleSelection(objects_dict, vis, main_window):
    for object_id, obj_info in objects_dict.items():
        if object_id == 'original':  # Skip the 'original' entry
             continue
        if obj_info.get('highlighted', False):  # Check if the 'highlighted' key exists and is True
            obj_info['selected'] = True
            obj = obj_info['object']  # Correctly reference the Open3D object
            obj.paint_uniform_color([0.540, 0.68, 0.52])  # Paint the object darker green
            vis.update_geometry(obj)
            print(f"Object {object_id} selected")
            main_window.update_text(f"Object selected!")

            break  # Exit the loop once an object is marked as selected


def handleDeselection(objects_dict, vis, main_window):
    for object_id, obj_info in objects_dict.items():
        if object_id == 'original':  # Skip the 'original' entry
            del objects_dict['original']
            continue
        if obj_info.get('selected', False):  # Check if the 'selected' key exists and is True
            obj_info['selected'] = False  # Mark the object as deselected
            obj_info['highlighted'] = False  # Mark the object as deselected
            obj = obj_info['object']  # Correctly reference the Open3D object
            obj.paint_uniform_color([0.5, 0.5, 0.5])  # Reset the object color to grey
            vis.update_geometry(obj)
            print(f"Object {object_id} deselected")
            main_window.update_text(f"Object {object_id} deselected")
            
            break  # Exit the loop once an object is marked as deselected


def handle_commands(clientSocket, vis, view_control, camera_parameters, geometry_dir, history, objects_dict, ls_dict, counters, main_window):
    try:
        # Attempt to receive data, but don't block indefinitely
        clientSocket.settimeout(0.1)  # Non-blocking with timeout
        command = getTCPData(clientSocket)
        if command:
            # Parse and handle the command
            parseCommand(command, view_control, camera_parameters, vis, geometry_dir, history, objects_dict, ls_dict, counters, main_window)
            vis.poll_events()
            vis.update_renderer()
    except s.timeout:
        # Ignore timeout exceptions, which are expected due to non-blocking call
        pass
  
    finally:
        clientSocket.settimeout(None)  # Reset to blocking mode




# ----------------------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------------------


def main():
    global marker
    
    app = QtWidgets.QApplication(sys.argv)
    
    stylesheet = """
        QMainWindow {
            background-color: #f2f2f2;
            font-family: Arial;
        }
        QLabel {
            color: #333;
            font-size: 20px;
        }
        /* Add more styles for other widgets */
        """
    app.setStyleSheet(stylesheet)
    serverSocket = startServer()
    clientSocket = makeConnection(serverSocket);


   # clientSocket = startClient() #for fake TCP server
    camera_parameters = o3d.camera.PinholeCameraParameters()

    width = 1280
    height = 960
    focal = 0.9616278814278851

    K = [[focal * width, 0, width / 2], [0, focal * width, height / 2], [0, 0, 1]]

    camera_parameters.extrinsic = np.array(
        [
            [1.204026265313826, 0, 0, -0.3759973645485034],
            [
                0,
                -0.397051999192357,
                0,
                4.813624436153903,
            ],
            [0, 0, 0.5367143925232766, 7.872266818189111],
            [0, 0, 0, 1],
        ]
    )

    camera_parameters.intrinsic.set_intrinsics(
        width=width, height=height, fx=K[0][0], fy=K[1][1], cx=K[0][2], cy=K[1][2]
    )
    o3d.t
    print("Testing mesh in Open3D...")

    mesh = o3d.geometry.TriangleMesh.create_box(width=0.2, height=0.2, depth=0.2)
    mesh.compute_vertex_normals()




    mesh2 = o3d.geometry.TriangleMesh.create_box(width=0.2, height=0.4, depth=0.2)
    mesh2.compute_vertex_normals()
    mesh2.translate(np.array([0.3, 0.5, 0.3]))
    
    
    mesh3 = o3d.geometry.TriangleMesh.create_box(width=0.4, height=0.1, depth=0.2)
    mesh3.compute_vertex_normals()
    mesh3.translate(np.array([0.2, 0.1, 0.4]))
    
    
    
    
    
    
    
    
    mesh.paint_uniform_color([0.5, 0.5, 0.5]) 
    mesh2.paint_uniform_color([0.5, 0.5, 0.5]) 
    mesh3.paint_uniform_color([0.5, 0.5, 0.5]) 

    
    
    

    # create visualizer and window.
    vis = o3d.visualization.Visualizer()
    

    vis.create_window(
        window_name="Open3D", width=width, height=height, left=50, top=50, visible=True
    )
    vis.add_geometry(mesh)
    vis.add_geometry(mesh2)
    vis.add_geometry(mesh3)

    vis.get_render_option().background_color = np.array([0.25, 0.25, 0.25])
    grid = create_grid(size=15, n=20, plane='xz', color=[0.5, 0.5, 0.5])

    # Add the grid to the visualizer
    vis.add_geometry(grid)


    
    #setup camera draw distance
    camera = vis.get_view_control()
    camera.set_constant_z_far(4500)

    objects_dict['object_1'] = {'object': mesh, 'center': mesh.get_center(), 'highlighted' : False, 'selected' : False,  'scale' : 100}
    objects_dict['object_2'] = {'object': mesh2, 'center': mesh2.get_center(), 'highlighted' : False, 'selected' : False, 'scale' : 100}
    objects_dict['object_3'] = {'object': mesh3, 'center': mesh3.get_center(), 'highlighted' : False, 'selected' : False, 'scale' : 100}


    ls_dict = {}

    counters = {"ls":0, "pcd":0, "meshes":0}


    view_control = vis.get_view_control()

    # Initialize required dictionaries and parameters
    geometry_dir = {"counters": {"pcd": 0, "ls": 0, "mesh": 0}}
    history = {"operation": "", "axis": "", "lastVal": "", 'last_extrusion_distance_x': 0.0,'last_extrusion_distance_y': 0.0, 'total_extrusion': 0.0}

    main_window = MainWindow(vis)
    main_window.show()

    # Setup a QTimer to periodically check for new commands
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: handle_commands(clientSocket, vis, view_control, camera_parameters, geometry_dir, history, objects_dict, ls_dict, counters, main_window))
    timer.start(25)  # Check every 100 milliseconds

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
