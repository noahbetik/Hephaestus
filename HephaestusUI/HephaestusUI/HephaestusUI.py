import open3d as o3d
import win32gui
import numpy as np
import keyboard
import socket as s
import time
import sys
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtGui import QFont

from camera_configs import predefined_extrinsics



# ----------------------------------------------------------------------------------------
# SHORTCUTS
# ----------------------------------------------------------------------------------------
#
# Ctrl + K + C --> comment block
#
# ----------------------------------------------------------------------------------------
# GLOBAL VARIABLES
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------------------------


objects_dict = {}
curr_highlighted = False
prevRotated = False
prevSnapped = False
marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
previous_look_at_point = None







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




def snap_to_closest_plane(vis, view_control):
    global prevSnapped
    
    # if (prevSnapped): 
    #     snap_isometric(vis, view_control) 
    #     return
    cam_params = view_control.convert_to_pinhole_camera_parameters()
    current_extrinsic = cam_params.extrinsic

    #extract rotational portion of extrinsic
    current_rotation = current_extrinsic[:3, :3]
    
    #find closest match
    closest_match = None
    smallest_difference = np.inf
    print("************current extrisnic: ", current_extrinsic)


    for name, rotation in predefined_extrinsics.items():
        difference = np.sum(np.abs(current_rotation - rotation))
        if difference < smallest_difference:
            closest_match = name
            smallest_difference = difference
            
    print(f"************Closest match: {closest_match}")
    
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


def move_camera_v3(view_control, vals):
    cam_params = view_control.convert_to_pinhole_camera_parameters()
    extrinsic = np.array(cam_params.extrinsic)

    # x axis handled by zoom
    extrinsic[0, 3] += vals[0] # y axis
    extrinsic[1, 3] += vals[1] # z axis

    cam_params.extrinsic = extrinsic
    view_control.convert_from_pinhole_camera_parameters(cam_params, True)


def rotate_camera(view_control, axis, degrees=5):
    cam_params = view_control.convert_to_pinhole_camera_parameters()
    R = cam_params.extrinsic[:3, :3]
    t = cam_params.extrinsic[:3, 3].copy()
    angle = np.radians(degrees)

    if axis == "y":
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
        sock.setblocking(True)  # Revert to the original blocking mode

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


# ----------------------------------------------------------------------------------------
# PARSE COMMAND
# ----------------------------------------------------------------------------------------


def parseCommand(
    command, view_control, camera_parameters, vis, geometry_dir, history, objects_dict, main_window
):
    # FORMAT
    # [command] [subcommand]

    global prevRotated
    info = command.split(" ")
    print(info)
    objectHandle = ""
    

    # Check for selected objects in objects_dict and update objectHandle to point to the mesh if selected
    for object_id, obj_info in objects_dict.items():
        if obj_info.get('selected', False):  # Check if the 'selected' key exists and is True
            objectHandle = obj_info['object']  # Now, objectHandle directly references the mesh object
            print(f"Selected object: {object_id}")
            break  # Assume only one object can be selected at a time; break after finding the first selected object
     # Assume only one object can be selected at a time; break after finding the first selected object

    if len(info) > 1:
        main_window.update_text(info[1])  # Use the update_text method of the main window

    match info[0]:
        case "motion":
            if objectHandle == "":
                handleCam(info[1:], view_control, history, vis)
                return ""
            else:
                # if (prevRotated) : snap_to_closest_plane(vis, view_control)
                # prevRotated = False
                handleUpdateGeo(info[1:], history, objectHandle, vis)
                return ""
        case "select":
            return handleSelection(objects_dict, vis, main_window)  # Assume this function handles object selection
        case "deselect":
            return handleDeselection(objects_dict, vis, main_window)  # Assume this function handles object selection
        case "create":
            handleNewGeo(info[1:], view_control, camera_parameters, vis, objects_dict)
            return ""
        case "update":
            # if objectHandle:
            #     if (prevRotated) : snap_to_closest_plane(vis, view_control)
            #     prevRotated = False
                handleUpdateGeo(info[1:], history, objectHandle, vis)
        case "home":
            snap_isometric(vis, view_control)
        case "snap":
            snap_to_closest_plane(vis, view_control)




def get_camera_look_at_position(view_control):
    cam_params = view_control.convert_to_pinhole_camera_parameters()
    
    # The inverse of the view matrix gives the camera's orientation and position in world space
    inverse_view_matrix = np.linalg.inv(cam_params.extrinsic)
    
    # The camera's position is the fourth column of the inverse view matrix
    camera_position = inverse_view_matrix[:3, 3]
    
    # The forward direction vector should be the negative of the third column of the rotation matrix
    forward_direction = -inverse_view_matrix[:3, 2]
    
    # Normalize the forward direction
    forward_direction_normalized = forward_direction / np.linalg.norm(forward_direction)
    
    # Define a distance in front of the camera to place the "look-at" point
    distance = 0.1  # Adjust as needed for your application
    
    # Calculate the look-at point by moving along the camera's forward direction from its position
    look_at_point = camera_position + distance * forward_direction_normalized
    
    look_at_point[2] = 0.5
    print(f"Look-at point: {look_at_point}")
    print(f"Camera position: {camera_position}")
    print(f"Forward direction: {forward_direction_normalized}")
    
    return look_at_point




def get_camera_look_at_position_v2(view_control):
    cam_params = view_control.convert_to_pinhole_camera_parameters()
    inverse_view_matrix = np.linalg.inv(cam_params.extrinsic)
    forward_direction = -inverse_view_matrix[:3, 2]
    camera_position = inverse_view_matrix[:3, 3]
    distance = 0.1
    look_at_point = camera_position + (forward_direction * distance)
    return look_at_point

def update_marker_position(marker, current_look_at_point):
    global previous_look_at_point

    # Ensure the previous look_at_point is not None
    if previous_look_at_point is not None:
        # Calculate the translation delta
        translation_delta = current_look_at_point - previous_look_at_point

        # Get the vertices of the marker's mesh
        vertices = np.asarray(marker.vertices)

        # Apply the translation delta to each vertex
        new_vertices = vertices + translation_delta
        marker.vertices = o3d.utility.Vector3dVector(new_vertices)

        # Recompute normals in case lighting effects are being used
        marker.compute_vertex_normals()
    else:
        # If there's no previous point (i.e., the first frame), don't move the marker
        pass

    # Update the global or class attribute to the current look_at_point for the next iteration
    previous_look_at_point = current_look_at_point

def highlight_object(geometry, color=[1, 0, 0]):  # Default highlight color is red
    geometry.paint_uniform_color(color)

def compute_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def highlight_objects_near_camera(vis, view_control, objects_dict):
    global marker
    # Get the camera position from the view control
    cam_params = view_control.convert_to_pinhole_camera_parameters()
    camera_position = np.asarray(cam_params.extrinsic[:3, 3])
    camera_position = camera_position + np.array([0, -0.1, 0])
    
    # update_marker_position(marker,camera_position)
    # vis.update_geometry(marker)

    # Initialize the closest distance and the object
    closest_distance = np.inf
    closest_object_id = None

    # Find the object closest to the camera
    for object_id, info in objects_dict.items():
        obj = info['object']
        centroid = info['center']

        distance = compute_distance(camera_position, centroid)
        
        if distance < closest_distance:
            closest_distance = distance
            closest_object_id = object_id

    # Highlight the closest object and unhighlight others
    for object_id, info in objects_dict.items():
        obj = info['object']
        if object_id == closest_object_id:
            # Highlight the closest object
            highlight_object(obj, color=[0.640, 0.91, 0.62])  # Light green for highlighted object
            info['highlighted'] = True
        else:
            # Unhighlight all other objects by resetting their color
            highlight_object(obj, color=[0.5, 0.5, 0.5])  # Default color
            info['highlighted'] = False
        
        # Assuming vis.update_geometry(obj) will reflect the changes in the visualization
        vis.update_geometry(obj)





def removeGeometry(vis, object):
    vis.remove_geometry(object)
    return



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
                    prevRotated = True
                    print("camera rotate update")
                    delta = float(subcommand[2]) - float(history["lastVal"])
                    rotate_camera(view_control, history["axis"], degrees=delta * alphaR)

                case "zoom":
                    print("camera zoom update")
                    delta = float(subcommand[2]) - float(history["lastVal"])
                    move_camera_v2(view_control, "x", delta * alphaZ)
            history["lastVal"] = subcommand[2]
        case _:
            print("INVALID COMMAND")


def handleNewGeo(subcommand, view_control, camera_parameters, vis, geometry_dir):
    alphaL = 0.001 # line scaling factor (maybe better way to do this)

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
            vis.add_geometry(new_box)
            name = "mesh" + str(geometry_dir["counters"]["mesh"])
            geometry_dir[name] = new_box
            geometry_dir["counters"]["mesh"] += 1

            # Set back the stored view matrix
            camera_parameters.extrinsic = current_view_matrix
            view_control.convert_from_pinhole_camera_parameters(camera_parameters, True)
        case "line":  # line handling not fully implemented yet
          #  print("**********SUBCOMMAND IS ",subcommand[1])
          #currently broken right now because ML side is not sending the start command
            if subcommand[1] == "start":
                pcd = o3d.geometry.PointCloud()
                ls = o3d.geometry.LineSet()

                print("Creating new line with endpoints ___ and ___")
                coords1 = [float(val)*alphaL for val in subcommand[2].strip("()").split(",")] + [0.0] # TODO: figure out which plane sketching on and place 0 accordingly
                coords2 = [float(val)*alphaL for val in subcommand[3].strip("()").split(",")] + [0.0] # TODO: figure out which plane sketching on and place 0 accordingly
                points = np.array([coords1, coords2]) 
                lines = np.array([[0, 1]])

                print(points)
                print(type(points))
                print(lines)
                print(type(lines))

                # Store the current view matrix
                current_view_matrix = (
                    view_control.convert_to_pinhole_camera_parameters().extrinsic
                )

                pcd.points = o3d.utility.Vector3dVector(points)

                ls.points = o3d.utility.Vector3dVector(points)
                ls.lines = o3d.utility.Vector2iVector(lines)

                vis.add_geometry(pcd)
                vis.add_geometry(ls)

                lsName = "ls" + str(geometry_dir["counters"]["ls"])
                geometry_dir[lsName] = ls
                pcdName = "pcd" + str(geometry_dir["counters"]["pcd"])
                geometry_dir[pcdName] = pcd
                geometry_dir["counters"]["ls"] += 1
                geometry_dir["counters"]["pcd"] += 1

                # Set back the stored view matrix
                camera_parameters.extrinsic = current_view_matrix
                view_control.convert_from_pinhole_camera_parameters(
                    camera_parameters, True
                )

            elif subcommand[1] == "end":
                ls_id = "ls" + str(geometry_dir["counters"]["ls"] - 1)
                pcd_id = "pcd" + str(geometry_dir["counters"]["pcd"] - 1)
                ls = geometry_dir[ls_id]
                pcd = geometry_dir[pcd_id]
                print("Ending line")
                # threshold for connecting closed-loop geometry
                all_points = np.asarray(pcd.points).tolist()
                print(pcd.points)
                print(all_points)
                if smartConnectBool(all_points[-1], all_points[0]):
                    ls.lines.extend(o3d.utility.Vector2iVector(np.array([[len(pcd.points), 0]])))

                vis.update_geometry(pcd)
                vis.update_geometry(ls)

            else:
                print("still sketching")
                ls_id = "ls" + str(geometry_dir["counters"]["ls"] - 1)
                pcd_id = "pcd" + str(geometry_dir["counters"]["pcd"] - 1)
                ls = geometry_dir[ls_id]
                pcd = geometry_dir[pcd_id]
                new_points = [float(val)*alphaL for val in subcommand[1].strip("()").split(",")] + [0.0]
                print(new_points)
                add_this = o3d.utility.Vector3dVector(
                    np.array([new_points])
                )
                pcd.points.extend(add_this)  # get proper references for this
                ls.points.extend(add_this)  # get proper references for this
                ls.lines.extend(o3d.utility.Vector2iVector(np.array([[len(pcd.points) - 1, len(pcd.points)]])))

                vis.update_geometry(pcd)
                vis.update_geometry(ls)

def handleUpdateGeo(subcommand, history, objectHandle, vis):
    alphaM = 0.01  # Translation scaling factor
    alphaR = 1  # Rotation scaling factor (in radians for Open3D)
    alphaS = 1  # Scaling scaling factor

    match subcommand[1]:
        case "start":
            print("Starting motion")
            history["operation"] = subcommand[0]  # Operation type
            history["axis"] = subcommand[2] if len(subcommand) > 2 else None  # Axis, if applicable
            history["lastVal"] = subcommand[3] if len(subcommand) > 3 else None  # Starting value, if applicable
        case "end":
            print("Ending motion")
            # Reset the history after operation ends
            history["operation"] = ""
            history["axis"] = ""
            history["lastVal"] = ""
            return
        case "position":
            if objectHandle:
                print("Updating position or transformation")
                # Assuming subcommand[2] is something like "(395,166)"
                try:
                    # Extract numerical values from the command
                    # This splits the string by comma after removing parentheses, then converts each part to float
                    coords = subcommand[2].strip("()").split(",")
                    currentX = float(coords[0])
                    currentY = float(coords[1])
                    
                    oldCoords = history["lastVal"].strip("()").split(",")
                    
                    oldX = float(oldCoords[0])
                    oldY = float(oldCoords[1])
                    

                    
                    deltaX = (currentX - oldX) * alphaM
                    deltaY = (currentY - oldY) * alphaM
                   # Retrieve the camera's rotation matrix
                    view_control = vis.get_view_control()
                    cam_params = view_control.convert_to_pinhole_camera_parameters()
                    # For Open3D, the view matrix is cam_params.extrinsic, which includes rotation and translation
                    # We only need the rotation part here
                    rotation_matrix = np.linalg.inv(cam_params.extrinsic[:3, :3])

                    # Transform the translation vector from view space to world space
                    # Note: deltaY is negated to account for screen-to-world coordinate system change
                    view_space_translation = np.array([deltaX, deltaY, 0])
                    world_space_translation = rotation_matrix.dot(view_space_translation)
                    
                    # Apply the world-space translation to the object
                    objectHandle.translate(world_space_translation, relative=True)
                    print("Translating object by", world_space_translation)
                    
                    # Update history with the current values for continuous operations
                    history["lastX"] = currentX
                    history["lastY"] = currentY
                    vis.update_geometry(objectHandle)
                except Exception as e:
                    print(f"Error processing numerical values from command: {e}")
                    
            history["lastVal"] = subcommand[2]

        case _:
            print("Invalid command")



def handleSelection(objects_dict, vis, main_window):
    for object_id, obj_info in objects_dict.items():
        if obj_info.get('highlighted', False):  # Check if the 'highlighted' key exists and is True
            obj_info['selected'] = True
            obj = obj_info['object']  # Correctly reference the Open3D object
            obj.paint_uniform_color([0.540, 0.68, 0.52])  # Paint the object darker green
            vis.update_geometry(obj)
            print(f"Object {object_id} selected")
            main_window.update_text(f"Object {object_id} selected")

            break  # Exit the loop once an object is marked as selected


def handleDeselection(objects_dict, vis, main_window):
    for object_id, obj_info in objects_dict.items():
        if obj_info.get('selected', False):  # Check if the 'selected' key exists and is True
            obj_info['selected'] = False  # Mark the object as deselected
            obj_info['highlighted'] = False  # Mark the object as deselected
            obj = obj_info['object']  # Correctly reference the Open3D object
            obj.paint_uniform_color([0.5, 0.5, 0.5])  # Reset the object color to grey
            vis.update_geometry(obj)
            print(f"Object {object_id} deselected")
            main_window.update_text(f"Object {object_id} deselected")
            
            break  # Exit the loop once an object is marked as deselected


def handle_commands(clientSocket, vis, view_control, camera_parameters, geometry_dir, history, objects_dict, main_window):
    try:
        # Attempt to receive data, but don't block indefinitely
        clientSocket.settimeout(0.1)  # Non-blocking with timeout
        command = getTCPData(clientSocket)
        if command:
            # Parse and handle the command
            parseCommand(command, view_control, camera_parameters, vis, geometry_dir, history, objects_dict, main_window)
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

    print("Testing mesh in Open3D...")

    mesh = o3d.geometry.TriangleMesh.create_box(width=0.2, height=0.2, depth=0.2)
    mesh.compute_vertex_normals()


    mesh2 = o3d.geometry.TriangleMesh.create_box(width=0.2, height=0.4, depth=0.2)
    mesh2.compute_vertex_normals()
    mesh2.translate(np.array([0.3, 0.5, 0.3]))

    # create visualizer and window.
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Open3D", width=width, height=height, left=50, top=50, visible=True
    )
    vis.add_geometry(mesh)
    vis.add_geometry(mesh2)
    vis.get_render_option().background_color = np.array([0.8, 0.9, 1.0])

    # marker_color = [1, 0, 0]  # Red color for the marker
    # marker.paint_uniform_color(marker_color)  # Color the marker

    # vis.add_geometry(marker)


    objects_dict['object_1'] = {'object': mesh, 'center': mesh.get_center(), 'highlighted' : False, 'selected' : False}
    objects_dict['object_2'] = {'object': mesh2, 'center': mesh2.get_center(), 'highlighted' : False, 'selected' : False}



    view_control = vis.get_view_control()

    # Initialize required dictionaries and parameters
    geometry_dir = {"counters": {"pcd": 0, "ls": 0, "mesh": 0}}
    history = {"operation": "", "axis": "", "lastVal": ""}

    main_window = MainWindow(vis)
    main_window.show()

    # Setup a QTimer to periodically check for new commands
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: handle_commands(clientSocket, vis, view_control, camera_parameters, geometry_dir, history, objects_dict, main_window))
    timer.start(1)  # Check every 100 milliseconds

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
