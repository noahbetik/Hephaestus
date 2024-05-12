import open3d as o3d
import win32gui
import numpy as np
import socket as s
import time
import sys

from csv import list_dialects

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtCore import QCoreApplication

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

objects_dict = {}
ls_dict = {}

curr_highlighted = False
prev_rotated = True
prev_added = False

extrusion_distance = 0
#counts the # of frames the user has held the 'delete' commmand
delete_count = 0 

rst_bit = 0
selected_pkt = 0
curr_pkt = 0
tcp_command_buffer = ""

regular_color = [0.5, 0.5, 0.5]
closest_color = [0.0, 0.482, 1.0]
selected_color = [0.157, 0.655, 0.271]
background_color = [0.11, 0.11, 0.11]
grid_color = [0.29, 0.29, 0.29]

# ----------------------------------------------------------------------------------------
# CLASSES
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

    def closeEvent(self, event): # is this used at all?
        super(Open3DVisualizerWidget, self).closeEvent(event)
        self.vis.destroy_window()

class TextDisplayWidget(QtWidgets.QLabel):
    def __init__(self, text="Welcome to Hephaestus", parent=None):
        super(TextDisplayWidget, self).__init__(parent)
        # Update font to a more modern one, such as "Segoe UI" and increase readability
        font = QtGui.QFont("Roboto", 24)  
        self.setFont(font)
        self.setText(text)
        self.setAlignment(QtCore.Qt.AlignCenter)  # Center align the text
        self.setStyleSheet("""
            background-color: transparent;
            color: white;
        """)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, vis):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Open3D and Text Display in PySide6")
        self.setGeometry(100, 100, 1280, 960)
        self.setStyleSheet("""
        QMainWindow {
            background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                              stop:0 #141414, stop:1 #141414);
        }
        """)

        # Create a central widget and set a layout for it
        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)
        vertical_layout = QtWidgets.QVBoxLayout(central_widget)

        # Horizontal layout for text and button
        text_layout = QtWidgets.QHBoxLayout()

        # Mode text widget, align it to the left
        self.mode_text_widget = QtWidgets.QLabel("Mode: Camera", self)
        self.mode_text_widget.setStyleSheet("""
            background-color: transparent;
            color: white;
            font: 18pt "Roboto";
        """)
        text_layout.addWidget(self.mode_text_widget, alignment=QtCore.Qt.AlignLeft)

        # Spacer to push dynamic text to center
        text_layout.addStretch()

        # Dynamic text widget (centered in the window)
        self.dynamic_text_widget = TextDisplayWidget("Waiting for command...", self)
        self.dynamic_text_widget.setStyleSheet("""
            background-color: transparent;
            color: white;
            font: 20pt "Roboto";
        """)
        text_layout.addWidget(self.dynamic_text_widget, alignment=QtCore.Qt.AlignCenter)

        # Spacer before the button to keep the text centered
        text_layout.addStretch()
        
        #set visualizer
        self.vis = vis

        # Button (right-aligned)
        self.action_button = QtWidgets.QPushButton('Reset', self)
        self.action_button.setStyleSheet("""
            QPushButton {
                font: 18pt 'Roboto';
                color: #FFFFFF; /* White text */
                background-color: #333333; /* Dark background */
                border: 2px solid #505050; /* Slightly lighter border for some contrast */
                border-radius: 10px; /* Curved borders */
                padding: 5px; /* Add some padding for aesthetics */
            }
            QPushButton:hover {
                background-color: #555555; /* Slightly lighter background on hover for visual feedback */
            }
        """)
        text_layout.addWidget(self.action_button, alignment=QtCore.Qt.AlignRight)
        self.action_button.clicked.connect(self.on_action_button_clicked)

        # Add the horizontal layout to the main vertical layout
        vertical_layout.addLayout(text_layout)

        # Create and add the Open3D visualizer widget
        self.open3d_widget = Open3DVisualizerWidget(vis, self)
        vertical_layout.addWidget(self.open3d_widget, 1)

        self.progress_bar = QtWidgets.QProgressBar(self)
        self.progress_bar.setMaximum(8)  # Set the maximum value to 20
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #505050; /* Darker border */
                border-radius: 5px;
                text-align: right; /* Aligns text to the right - note this won't move the text out of the bar */
                background-color: #333333; /* Dark background */
                color: #FFFFFF; /* Light text color */
                min-height: 10px; /* Makes the progress bar thinner */
                max-height: 10px; /* Ensures the height does not exceed this value */

            }

            QProgressBar::chunk {
                background-color: #32CD32; /* Green */
                border-radius: 4px; /* Optional: Matches the bar's border-radius for consistency */
            }
        """)

        vertical_layout.addWidget(self.progress_bar)
        # Add Full-Screen Toggle Button
        self.fullscreen_button = QtWidgets.QPushButton('Full Screen', self)
        self.fullscreen_button.setStyleSheet("""
            QPushButton {
                font: 18pt 'Roboto';
                color: #FFFFFF; /* White text */
                background-color: #333333; /* Dark background */
                border: 2px solid #505050; /* Slightly lighter border for some contrast */
                border-radius: 10px; /* Curved borders */
                padding: 5px; /* Add some padding for aesthetics */
            }
            QPushButton:hover {
                background-color: #555555; /* Slightly lighter background on hover for visual feedback */
            }
        """)
        text_layout.addWidget(self.fullscreen_button, alignment=QtCore.Qt.AlignRight)
        self.fullscreen_button.clicked.connect(self.toggle_full_screen)



    def update_progress(self, value):
        # Update the progress bar with the new value
        self.progress_bar.setValue(value)

    def toggle_full_screen(self):
        if self.isFullScreen():
            self.showNormal()  # If the window is in full-screen mode, exit full-screen
            self.fullscreen_button.setText('Full Screen')  # Update button text
        else:
            self.showFullScreen()  # Enter full-screen mode
            self.fullscreen_button.setText('Exit Full Screen')  # Update button text


    def on_action_button_clicked(self):
        # This method will be called when the button is clicked
        global objects_dict, ls_dict, curr_highlighted, prev_rotated, prev_added, extrusion_distance, delete_count, rst_bit

        print("Reset button pressed!")
        self.update_dynamic_text("Welcome to Hephaestus!")
        curr_highlighted = False
        prev_rotated = True
        prev_added = False
        extrusion_distance = 0
        delete_count = 0
        rst_bit = 1

        # Delete all meshes and sketches (respectively) & clear out dictionaries that store references to them

        for obj in objects_dict.values():
            self.vis.remove_geometry(obj['object'], reset_bounding_box=False)
        objects_dict = {}

        for k, v in ls_dict.items():
            self.vis.remove_geometry(ls_dict[k], reset_bounding_box=False)
        ls_dict = {}
        
        # Snap back to isometric view

        snap_isometric(self.vis, self.vis.get_view_control(), steps = 105)
        self.update_progress(0)
        

    def update_dynamic_text(self, new_text):
        self.dynamic_text_widget.setText(new_text)


    def update_mode_text(self, new_text):
        self.mode_text_widget.setText(f"Mode: {new_text}")


# ----------------------------------------------------------------------------------------
# PLANE AND CAMERA FUNCTIONS
# ----------------------------------------------------------------------------------------


def create_grid(size=10, n=10, plane='xz', color=regular_color):
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
    else:
        print("Grid lines are only supported on the xz plane!")
        # could change this later if we want 3D grid, just need to replicate logic for xy and yz planes
        sys.exit()

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

    closest_match = None
    smallest_difference = np.inf

    # Find closest match
    for name, rotation in extrinsics.items():
        difference = np.sum(np.abs(current_rotation - rotation))
        if difference < smallest_difference:
            closest_match = name
            smallest_difference = difference
    
    return closest_match

def identify_plane(current_extrinsic):
    
    # Identify closest major plane first to find in/out axis  
    closest_plane = closest_config(current_extrinsic)

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
        if np.allclose(np.dot(current_extrinsic, [0, 0, -1]), direction):
            return axis


def snap_to_closest_plane(vis, view_control):

    # Obtain the current extrinsic parameters and find closest plane 
    cam_params = view_control.convert_to_pinhole_camera_parameters()
    current_extrinsic = cam_params.extrinsic
    closest_match = closest_config(current_extrinsic)
    
    # Update extrinsic to closest plane and move camera
    updated_extrinsic = current_extrinsic.copy()
    updated_extrinsic[:3, :3] = predefined_extrinsics[closest_match]
    smooth_transition(vis, view_control, updated_extrinsic)


def snap_isometric(vis, view_control, steps = 85):
    # Obtain the current extrinsic parameters
    cam_params = view_control.convert_to_pinhole_camera_parameters()

    # Predefined extrinsic set for isometric view
    target_extrinsic = np.array([
        [ 8.68542871e-01, -1.11506045e-03,  4.95612791e-01,  1.71200000e-01],
        [-2.08451221e-01, -9.08069551e-01,  3.63259933e-01,  7.36100000e-02],
        [ 4.49645828e-01, -4.18817916e-01, -7.88929771e-01,  1.99985850e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
    ])
    
    # Smooth transition to isometric view
    smooth_transition(vis, view_control, target_extrinsic, steps)


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


def smooth_transition(vis, view_control, target_extrinsic, steps=85):
    
    # Obtain the current and target extrinsic parameters
    current_extrinsic = view_control.convert_to_pinhole_camera_parameters().extrinsic
    current_translation = current_extrinsic[:3, 3]
    target_translation = target_extrinsic[:3, 3]

    # Convert rotation matrices to quaternions
    current_quat = convert_rotation_matrix_to_quaternion(current_extrinsic[:3, :3])
    target_quat = convert_rotation_matrix_to_quaternion(target_extrinsic[:3, :3])

    for step in range(steps):
        fraction = step / float(steps)

        # Linear interpolation of translation
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

    
def move_camera(view_control, vals, threshold=0.01):
    """
    Moves the camera based on the provided values, ignoring movements that are below a specified threshold.
    
    Parameters:
    - view_control: The view control object to manipulate the camera's extrinsic parameters.
    - vals: A tuple or list with two elements indicating the amount of movement in the y and z axes, respectively.
    - threshold: The minimum movement required to apply the camera movement. Movements below this threshold are ignored.
    """
    
    cam_params = view_control.convert_to_pinhole_camera_parameters()
    extrinsic = np.array(cam_params.extrinsic)
    significant_movement = False
    
    # Check and apply movement for the x axis if it exceeds the threshold
    if abs(vals[0]) > threshold:
        extrinsic[2, 3] += vals[0]
        significant_movement = True

    # Check and apply movement for the y axis if it exceeds the threshold
    if abs(vals[1]) > threshold:
        extrinsic[0, 3] += vals[1]
        significant_movement = True

    # Check and apply movement for the z axis if it exceeds the threshold
    if abs(vals[2]) > threshold:
        extrinsic[1, 3] += vals[2]
        significant_movement = True

    # If any significant movement occurred, update the camera parameters
    if significant_movement:
        cam_params.extrinsic = extrinsic
        view_control.convert_from_pinhole_camera_parameters(cam_params, True)



def rotate_camera(view_control, axis, degrees=5):
    angle = np.radians(degrees)
    if abs(angle) < 0.01:
        return

    # Obtain current rotation matrix from extrinsics
    cam_params = view_control.convert_to_pinhole_camera_parameters()
    R = cam_params.extrinsic[:3, :3]
    t = cam_params.extrinsic[:3, 3].copy()
    angle = np.radians(degrees)

    # Apply rotation delta (in radians) to rotation matrix along correct axis
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

    # Set the new extrinsic matrix
    cam_params.extrinsic = new_extrinsic
    view_control.convert_from_pinhole_camera_parameters(cam_params, True)
    

# ----------------------------------------------------------------------------------------
# TCP FUNCTIONS
# ----------------------------------------------------------------------------------------


def start_server():
    server_socket = s.socket(s.AF_INET, s.SOCK_STREAM) # IPV4 address family, datastream connection
    server_addr = ('localhost', 4445) # (IP addr, port)
    server_socket.bind(server_addr)
    print("Server started on " + str(server_addr[0]) + " on port " + str(server_addr[1]))

    return server_socket

def make_connection(server_socket):
    server_socket.listen(1) # max number of queued connections
    print("Waiting for connection...")
    client_socket, client_addr = server_socket.accept()
    print("\nConnection from " + str(client_addr[0]) + " on port " + str(client_addr[1]))
    
    client_socket.setblocking(True)
    
    return client_socket


def get_tcp_data(sock, sketch):
    global tcp_command_buffer
    global selected_pkt, rst_bit, curr_pkt

    try:
        # Send reset
        if rst_bit == 1: 
            sock.sendall("RST".encode("ascii"))
            rst_bit = 0
        # Send select
        elif selected_pkt == 1:
            sock.sendall("SEL".encode("ascii"))
            curr_pkt = 1
        # Send deselect
        else:
            sock.sendall("DES".encode("ascii"))
            curr_pkt = 0

        data = sock.recv(64)  # Attempt to read more data

        # sock.setblocking(True) # moved to client socket creation bc never changed, but still paranoid

        # Decode and append to buffer
        tcp_command_buffer += data.decode('ascii')

        # Process if delimiter is found
        if '\n' in tcp_command_buffer:
            command, tcp_command_buffer = tcp_command_buffer.split('\n', 1)
            command = command.strip("$")  # Strip any '$' characters
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


def close_client(sock):
    sock.close()
    

# ----------------------------------------------------------------------------------------
# GEOMETRY FUNCTIONS
# ----------------------------------------------------------------------------------------

def vector_distance(p1, p2):
    # Extract coordinates
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    
    # Calculate the distance
    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) ** 0.5
    return distance


def linear_interpolate_3d(point1, point2, n):
    # Calculate the step size for each dimension
    step = [(point2[i] - point1[i]) / (n + 1) for i in range(3)]

    # Generate the intermediate points
    intermediate_points = []
    for i in range(1, n + 1):
        x = point1[0] + step[0] * i
        y = point1[1] + step[1] * i
        z = point1[2] + step[2] * i
        intermediate_points.append([x, y, z])

    return intermediate_points

def scale_polygon_2d(polygon, scale_factor):
    # Compute the centroid of the polygon
    centroid = np.mean(polygon, axis=0)

    # Translate the polygon to the origin
    translated_polygon = polygon - centroid

    # Scale the polygon
    scaled_polygon = translated_polygon * scale_factor

    # Translate the scaled polygon back to its original position
    scaled_polygon += centroid

    return scaled_polygon.tolist()


def smart_connect(end_point, start_point):
    # Absolute distance threshold for connection
    threshold = 0.15
    if (
        abs(end_point[0] - start_point[0]) < threshold
        and abs(end_point[1] - start_point[1]) < threshold
    ):
        return True
    else:
        return False


def sketch_extrude(counters, vis):
    
    global ls_dict
    global objects_dict

    # Get PointCloud and LineSet references 
    ls_id = "ls" + str(counters["ls"] -1)
    pcd_id = "pcd" + str(counters["pcd"] -1)
    ls = ls_dict[ls_id]
    pcd = ls_dict[pcd_id]

    # Tunable scaling factors
    scale_factor = 5
    stackFactor = 5
    stackHeight = 0.1
    stepSize = stackHeight / stackFactor

    points = np.asarray(pcd.points).tolist()

    # Increase point density of original sketch
    for p in range(len(points) - 1): 
        points = points + linear_interpolate_3d(points[p], points[p+1], scale_factor)
    points = points + linear_interpolate_3d(points[-1], points[0], scale_factor)

    stacked = []

    # Stack sketch outline to create volume
    for p in points:
        for i in np.arange(stepSize, 0.1, stepSize):
            stacked.append([p[0], p[1], i])

    # End case -- final layer of stack
    points2 = [[x, y, z+stackHeight] for x, y, z in points] 

    totalScaled = []

    # Fill in top and bottom faces
    for i in range (0, scale_factor):
        scaled1 = scale_polygon_2d(np.array(points), i/scale_factor)
        scaled2 = scale_polygon_2d(np.array(points2), i/scale_factor)
        
        totalScaled = totalScaled + scaled1 + scaled2

    # Update PointCloud with new points
    points = points + points2 + stacked + totalScaled
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    vis.update_geometry(pcd) # do this incrementally to visually show progress?

    # Set up for surface reconstruction
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50))
    pcd.orient_normals_consistent_tangent_plane(k=40)

    # Poisson surface reconstruction
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=12)

    # Paint the mesh to prevent it from being black
    mesh.paint_uniform_color([0.7, 0.7, 0.7])  # Light gray color
    mesh.compute_vertex_normals()

    # Remove LineSet and PoitnCloud geometry and visualize mesh
    vis.remove_geometry(ls)
    vis.remove_geometry(pcd)
    ls_dict = {}
    add_geometry(vis, mesh, objects_dict, "sketch", False)


# ----------------------------------------------------------------------------------------
# PARSE COMMAND
# ----------------------------------------------------------------------------------------

def parse_command(
    command, view_control, camera_parameters, vis, geometry_dir, history, objects_dict, counters, main_window
):
    # FORMAT
    # [command] [subcommand]

    global prev_rotated
    global delete_count
    global prev_added
    global selected_pkt
    global ls_dict
    global rst_bit
    
    info = command.split(" ")
    object_handle = ""    
    object_id = ""

    # Check for selected objects in objects_dict and update object_handle to point to the mesh if selected
    for id, obj_info in objects_dict.items():
        # Check if the 'selected' key exists and is True --> object_handle directly references the mesh object
        if obj_info.get('selected', False):
            object_handle = obj_info['object']  
            object_id = id
            break  
            # Assume only one object can be selected at a time; break after finding the first selected object

    match info[0]:
        case "motion":
            # Handle camera movement if no object currently selected and we are not extruding
            if object_handle == "" and info[1:][0] != "extrude":
                main_window.update_mode_text("Camera")
                handle_cam(info[1:], view_control, history, vis, main_window)
                return ""

            # Handle object movment if an object is currently selected
            elif object_handle:
                main_window.update_mode_text("Object")
                handle_update_geo(info[1:], history, object_handle, vis, main_window, objects_dict, object_id, ls_dict)
                object_handle.paint_uniform_color(selected_color)
                return ""
            
        case "select":
            if len(ls_dict) == 2:
                main_window.update_dynamic_text("Performing extrude operation. Please wait.")
                # Force the processing of the events
                QCoreApplication.processEvents()

                # Force deselect in case something is already selected
                camera_parameters = vis.get_view_control().convert_to_pinhole_camera_parameters()
                handle_deselection(objects_dict, vis, main_window)
                
                sketch_extrude(counters, vis) # NOAH DEBUG -- ??? don't understand control flow here

                smooth_transition(vis, view_control, np.array([
                [ 0.99994158, -0.00229353,  0.0105631 , -1.22174849],
                [-0.00238567, -0.99995914,  0.00871879,  0.39801206],
                [ 0.01054267, -0.00874348, -0.9999062 ,  1.79901982],
                [ 0.        ,  0.        ,  0.        ,  1.        ]
                ]), steps = 25)
                
                # Reset ML side as a precaution
                rst_bit = 1
            else:
                prev_added = False
                delete_count = 0 
                main_window.update_mode_text("Object")
                selected_pkt = handle_selection(objects_dict, vis, main_window)
                history['operation'] = "select"

        case "deselect":
            delete_count = 0
            selected_pkt  = 0

            main_window.update_mode_text("Camera")
            selected_pkt = handle_deselection(objects_dict, vis, main_window)
            history['operation'] = "deselect"

        case "create":
            delete_count = 0
            if not prev_added:
                handle_new_geo(info[1:], view_control, camera_parameters, vis, objects_dict, counters, main_window)
                handle_deselection(objects_dict, vis, main_window)
            return ""
        
        case "update":
            delete_count = 0
            handle_update_geo(info[1:], history, object_handle, vis, main_window, objects_dict, object_id, ls_dict)

        case "snap":
            if info[1:][0] == "iso":
                snap_isometric(vis, view_control)
            elif info[1:][0] == "home":
                snap_to_closest_plane(vis, view_control)
 
        case "delete":
            history['operation'] = "delete"
            delete_count += 1
            if (delete_count > 7 and object_handle): # why 7?
                remove_geometry(vis, object_handle, object_id)
                delete_count = 0

            # Priority to delete sketch first -- make sure sketch exists before trying to delete
            if len(ls_dict) >= 2: 
                ls_id = "ls" + str(counters["ls"] - 1)
                pcd_id = "pcd" + str(counters["pcd"] - 1)

                vis.remove_geometry(ls_dict[ls_id])
                vis.remove_geometry(ls_dict[pcd_id])
                ls_dict = {}
                
                smooth_transition(vis, view_control, np.array([
                    [ 0.99994158, -0.00229353,  0.0105631 , -1.22174849],
                    [-0.00238567, -0.99995914,  0.00871879,  0.39801206],
                    [ 0.01054267, -0.00874348, -0.9999062 ,  1.79901982],
                    [ 0.        ,  0.        ,  0.        ,  1.        ]
                    ]), steps = 25)

        
            try:
                if 'original' in objects_dict[object_id] and object_id:
                    history['total_extrusion_x'] = 0
                    history['total_extrusion_y'] = 0

                    vis.remove_geometry(object_handle, reset_bounding_box=False)
                    object_handle = clone_mesh(objects_dict[object_id]['original'])
                    object_handle.compute_vertex_normals()
                    object_handle.paint_uniform_color(selected_color)
                    vis.add_geometry(object_handle, reset_bounding_box=False)
                    vis.update_geometry(object_handle)

                    # Update the current object with the original
                    objects_dict[object_id]['object'] = object_handle  
                    history['last_extrusion_distance_x'] = 0.0
                    history['last_extrusion_distance_y'] = 0.0

                    objects_dict[object_id]['total_extrusion_x'] = 0.0
                    objects_dict[object_id]['total_extrusion_y'] = 0.0
                    objects_dict[object_id]['selected'] = True
            except KeyError:
                pass
        
        # Extra-long hold for delete compared to revert
        case "lock-in":
            if history['operation'] == "delete" and object_handle != "":
                main_window.update_dynamic_text("Object reverted. Long hold to delete object")
                main_window.update_progress(delete_count*1.4)
            elif history['operation'] == "select":
                pass
            else: main_window.update_progress(int(info[1])*1.75)
            
        case _:
            history["lastVal"] = info[1:][2]

            

def highlight_objects_near_camera(vis, view_control, objects_dict):
    
    # Get the camera position from the view control
    cam_params = view_control.convert_to_pinhole_camera_parameters()
    closest_match = closest_config(cam_params.extrinsic)
    extrinsic = predefined_extrinsics[closest_match]
    forward_vector = forward_vectors[closest_match]
    
    camera_position = np.array(cam_params.extrinsic[:3, 3]) 
    camera_position[0] *= forward_vector[0]
    camera_position[1] *= forward_vector[1]
    camera_position[2] *= forward_vector[2]

    # Initialize the closest distance and the object ID
    closest_distance = np.inf
    closest_object_id = None

    # Find the object closest to the camera
    for object_id, info in objects_dict.items():
        # Skip the 'original' entry
        if object_id == 'original':
            continue
        obj = info['object']
        centroid = info['center']

        distance = np.linalg.norm(camera_position[:2] - centroid[:2])       
        
        if distance < closest_distance:
            closest_distance = distance
            closest_object_id = object_id

    # Highlight the closest object and unhighlight others
    for object_id, info in objects_dict.items():
        # Skip the 'original' entry
        if object_id == 'original':  
            continue
        obj = info['object']
        if object_id == closest_object_id:
            # Highlight the closest object
            obj.paint_uniform_color(closest_color)
            info['highlighted'] = True
        else:
            # Unhighlight all other objects by resetting their color
            obj.paint_uniform_color(regular_color)
            info['highlighted'] = False
        
        # Update the geometry to reflect changes
        vis.update_geometry(obj)

    vis.update_renderer() # NOAH / DEREK -- check if strictly necessary


def remove_geometry(vis, obj, id):

    # Remove object from vis and remove reference from objects dictionary
    vis.remove_geometry(obj, reset_bounding_box = False)
    del objects_dict[id]


def add_geometry(vis, obj, objects_dict, objType, wasSpawned):
    
    # Generate a unique object ID based on the current number of items in objects_dict
    object_id = f"object_{len(objects_dict) + 1}"
    
    # Compute the center of the object for translation to the origin
    center = obj.get_center()

    # Update the object's color
    obj.paint_uniform_color([0.5, 0.5, 0.5])
    
    if wasSpawned:
        # Vector to move the object's center to the origin
        translation_vector = -center  
    
        # Translate the object to the origin & add to vis
        obj.translate(translation_vector, relative=False)
        obj.translate(np.array([0, 0.2, 0]))
        vis.add_geometry(obj, reset_bounding_box=False)
        
    else:
        # Add the object to the visualizer
        vis.add_geometry(obj, reset_bounding_box = False)

    # Add the new object to objects_dict with its properties
    objects_dict[object_id] = {
        'object': obj, 
        'center': center,
        'highlighted': False, 
        'selected': False, 
        'scale': 100,
        'type' : objType
    }


def scale_object(object_handle, delta, min_size=0.01, max_size=2):
    # Intended scale factor based on the delta
    scale_factor = 1 + delta
    
    # Get width, height, and depth of the object's bounding box
    bbox = object_handle.get_axis_aligned_bounding_box()
    extents = bbox.get_extent()  
    
    # Calculate the new extents after scaling
    new_extents = extents * scale_factor
    
    # Check if any of the new extents fall below the minimum size
    if np.any(new_extents < min_size):
        required_scale_factors_for_min = min_size / extents
        scale_factor = max(required_scale_factors_for_min.max(), scale_factor)

    # Check if any of the new extents exceed the maximum size
    elif np.any(new_extents > max_size):
        required_scale_factors_for_max = max_size / extents
        scale_factor = min(required_scale_factors_for_max.min(), scale_factor)
    
    # Compute the object's center for uniform scaling about its center
    center = bbox.get_center()
    
    # Apply the scaling
    object_handle.scale(scale_factor, center)



def rotate_object(object_handle, axis, degrees=5):
    
    # Calculate the rotation angle in radians
    angle = np.radians(degrees)
    if abs(angle) < 0.02:
        return
    
    # Compute the object's center
    center = object_handle.get_center()
    
    # Define the rotation matrix for each axis
    if axis == "x":
        rotation_matrix = np.array([
            [1, 0, 0, 0],
            [0, np.cos(angle), -np.sin(angle), 0],
            [0, np.sin(angle), np.cos(angle), 0],
            [0, 0, 0, 1]
        ])
        
    elif axis == "y":
        rotation_matrix = np.array([
            [np.cos(angle), 0, np.sin(angle), 0],
            [0, 1, 0, 0],
            [-np.sin(angle), 0, np.cos(angle), 0],
            [0, 0, 0, 1]
        ])
        
    elif axis == "z":
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0, 0],
            [np.sin(angle), np.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")
    
    # Translate the object to the origin, rotate, and then translate back
    translate_to_origin = np.eye(4)
    translate_to_origin[:3, 3] = -center
    translate_back = np.eye(4)
    translate_back[:3, 3] = center
    
    # Combine the transformations
    transformation = translate_back @ rotation_matrix @ translate_to_origin
    
    # Apply the transformation
    object_handle.transform(transformation)
    
    
def handle_cam(subcommand, view_control, history, vis, main_window):
    
    global prev_rotated
    # FORMAT:
    # start [operation] [axis] [position]
    # position n
    # position n+1
    # position n+2
    # ...
    # position n+m
    # end [operation] [axis]

    # history: {operation:operation, axis:axis, lastVal:lastVal}

    # Scaling factors for pan, rotate, zoom (respectively)
    alpha_m = 0.01
    alpha_r = 1
    alpha_z = -0.01

    match subcommand[1]:
        case "start":
            history["operation"] = subcommand[0] # NOAH/DEREK -- in theory don't need to store this anymore since optype sent each update
            if history["operation"] != "rotate":
                history["lastVal"] = subcommand[2]
            else:
                history["axis"] = subcommand[2]
                history["lastVal"] = subcommand[3]

        case "end":
            main_window.update_dynamic_text("Waiting for command...")
            history["operation"] = ""
            history["axis"] = ""
            history["lastVal"] = ""
            return
        
        case "position":
            match history["operation"]:
                case "pan":
                    main_window.update_dynamic_text("Panning camera")
                    
                    # Parse deltas and apply scaling factors
                    splitCoords = subcommand[2].strip("()").split(",")
                    oldCoords = history["lastVal"].strip("()").split(",")
                    delta_y = (float(splitCoords[0]) - float(oldCoords[0])) * alpha_m
                    delta_z = (float(splitCoords[1]) - float(oldCoords[1])) * alpha_m
                    move_camera(view_control, [0, delta_y, delta_z])

                    highlight_objects_near_camera(vis, view_control, objects_dict)
                    
                case "rotate":
                    main_window.update_dynamic_text("Rotating camera")
                    
                    # Parse delta and apply scaling factors
                    prev_rotated = True
                    delta = float(subcommand[2]) - float(history["lastVal"])
                    rotate_camera(view_control, history["axis"], degrees=delta * alpha_r)
                    
                    highlight_objects_near_camera(vis, view_control, objects_dict)

                case "zoom":
                    main_window.update_dynamic_text("Zooming camera")
                    
                    # Parse delta and apply scaling factors
                    delta = float(subcommand[2]) - float(history["lastVal"])
                    move_camera(view_control, [delta * alpha_z, 0, 0])
                    
                    highlight_objects_near_camera(vis, view_control, objects_dict)

            history["lastVal"] = subcommand[2]
            
        case _:
            print("INVALID COMMAND")


def handle_new_geo(subcommand, view_control, camera_parameters, vis, objects_dict, counters, main_window):
    global view_axis
    global prev_added
    global ls_dict
    global rst_bit

    # Line scaling factor
    alpha_l = 0.002 

    match subcommand[0]:
        case "cube":
            # Store the current view matrix
            current_view_matrix = view_control.convert_to_pinhole_camera_parameters().extrinsic
            
            # Create and add the cube
            new_box = o3d.geometry.TriangleMesh.create_box(width=0.2, height=0.2, depth=0.2)
            new_box.compute_vertex_normals()
            add_geometry(vis, new_box, objects_dict, "cube", True)
            prev_added = True
            main_window.update_dynamic_text("Cube added at origin")

        case "sphere":
            # Store the current view matrix
            current_view_matrix = view_control.convert_to_pinhole_camera_parameters().extrinsic
            
            # Create and add the sphere
            new_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
            new_sphere.compute_vertex_normals()
            add_geometry(vis, new_sphere, objects_dict, "sphere", True)
            prev_added = True
            main_window.update_dynamic_text("Sphere added at origin")

        case "triangle":
            # Store the current view matrix
            current_view_matrix = view_control.convert_to_pinhole_camera_parameters().extrinsic

            # scale_factor --> face size, depth --> prism length
            scale_factor = 2
            depth = 0.1
            vertices = np.array([
                [0, 0, -depth/2],  # Base center back
                [scale_factor * 0.1, 0, -depth/2],  # Base right back
                [scale_factor * 0.05, scale_factor * 0.1, -depth/2],  # Top back
                [0, 0, depth/2],  # Base center front
                [scale_factor * 0.1, 0, depth/2],  # Base right front
                [scale_factor * 0.05, scale_factor * 0.1, depth/2],  # Top front
            ])  # Triangle vertices

            # Define the triangles (faces of the 3D object)
            triangles = np.array([
                [0, 2, 1],  # Back face
                [3, 4, 5],  # Front face
                [0, 3, 5], [5, 2, 0],  # Left side faces
                [0, 1, 4], [4, 3, 0],  # Bottom side faces
                [1, 2, 5], [5, 4, 1],  # Right side faces
            ])  # Triangle indices

            new_triangle = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(vertices),
                                                    triangles=o3d.utility.Vector3iVector(triangles))
            new_triangle.compute_vertex_normals()
            add_geometry(vis, new_triangle, objects_dict, "triangle", True)
            prev_added = True
            main_window.update_dynamic_text("Triangle added at origin")

        case "line":
            main_window.update_dynamic_text("Drawing new line")

            # Store the current view matrix
            current_view_matrix = (
                view_control.convert_to_pinhole_camera_parameters().extrinsic
            )

            if subcommand[1] == "start":
                if len(ls_dict) > 0:
                    # Get references if sketch already exists (NOAH -- shouldn't exist tho???)
                    ls_id = "ls" + str(counters["ls"] - 1)
                    pcd_id = "pcd" + str(counters["pcd"] - 1)
                    ls = ls_dict[ls_id]
                    pcd = ls_dict[pcd_id]

                    vis.remove_geometry(ls)
                    vis.remove_geometry(pcd)
                    
                    ls_dict = {}
          

                smooth_transition(vis, view_control, np.array([
                [ 9.99986618e-01, -1.10921734e-03,  5.05311841e-03, -1.28562713e+00],
                [-1.13032407e-03, -9.99990642e-01,  4.17603074e-03,  4.10222709e-01],
                [ 5.04843899e-03, -4.18168652e-03, -9.99978513e-01,  2.46971612e+00],
                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
                ]))

                closest_plane = predefined_extrinsics[closest_config(current_view_matrix)]
                view_axis = identify_plane(closest_plane) # global var --> ???
                
                pcd = o3d.geometry.PointCloud()
                ls = o3d.geometry.LineSet()

                # Convert 2D coordinates given by ML into 3D coordinates
                # Depends on given axis (only +ve x implemented currently)
                match view_axis:
                    case 'x':
                        coords1 = [0.0] + [float(val)*alpha_l for val in subcommand[2].strip("()").split(",")]
                        coords2 = [0.0] + [float(val)*alpha_l for val in subcommand[3].strip("()").split(",")]
                    case '-x':
                        coords1 = [0.0] + [float(val)*alpha_l for val in subcommand[2].strip("()").split(",")]
                        coords2 = [0.0] + [float(val)*alpha_l for val in subcommand[3].strip("()").split(",")]
                    case 'y':
                        coords1 = [float(val)*alpha_l for val in subcommand[2].strip("()").split(",")].insert(1, [0.0])
                        coords2 = [float(val)*alpha_l for val in subcommand[3].strip("()").split(",")].insert(1, [0.0])
                    case '-y':
                        coords1 = [float(val)*alpha_l for val in subcommand[2].strip("()").split(",")].insert(1, [0.0])
                        coords2 = [float(val)*alpha_l for val in subcommand[3].strip("()").split(",")].insert(1, [0.0])
                    case 'z':
                        coords1 = [float(val)*alpha_l for val in subcommand[2].strip("()").split(",")] + [0.0]
                        coords2 = [float(val)*alpha_l for val in subcommand[3].strip("()").split(",")] + [0.0]
                    case '-z':
                        coords1 = [float(val)*alpha_l for val in subcommand[2].strip("()").split(",")] + [0.0]
                        coords2 = [float(val)*alpha_l for val in subcommand[3].strip("()").split(",")] + [0.0]

                points = np.array([coords1, coords2]) 
                lines = np.array([[0, 1]])

                # Store the current view matrix
                current_view_matrix = (
                    view_control.convert_to_pinhole_camera_parameters().extrinsic
                )

                # Add PointCloud and LineSet objects to vis
                pcd.points = o3d.utility.Vector3dVector(points)
                ls.points = o3d.utility.Vector3dVector(points)
                ls.lines = o3d.utility.Vector2iVector(lines)
                vis.add_geometry(pcd)
                vis.add_geometry(ls)

                # Store PointCloud and LineSet references
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
                main_window.update_dynamic_text("Waiting for command...")

                # For updating references
                ls_id = "ls" + str(counters["ls"] - 1)
                pcd_id = "pcd" + str(counters["pcd"] - 1)
                ls = ls_dict[ls_id]
                pcd = ls_dict[pcd_id]

                # threshold for connecting closed-loop geometry
                all_points = np.asarray(pcd.points).tolist()

                if smart_connect(all_points[-1], all_points[0]):
                    ls.lines.extend(o3d.utility.Vector2iVector(np.array([[len(pcd.points), 0]])))
                
                # Update references and vis
                ls_dict[ls_id] = ls
                ls_dict[pcd_id] = pcd
                vis.update_geometry(pcd)
                vis.update_geometry(ls)

                view_axis = ''

            else:
                # Update references
                ls_id = "ls" + str(counters["ls"] - 1)
                pcd_id = "pcd" + str(counters["pcd"] - 1)
                ls = ls_dict[ls_id]
                pcd = ls_dict[pcd_id]

                # Convert 2D coordinates given by ML into 3D coordinates
                # Depends on given axis (only +ve x implemented currently)
                match view_axis:
                    case 'x':
                        new_points = [0.0] + [float(val)*alpha_l for val in subcommand[1].strip("()").split(",")]
                    case '-x':
                        new_points = [0.0] + [float(val)*alpha_l for val in subcommand[1].strip("()").split(",")]
                    case 'y':
                        new_points = [float(val)*alpha_l for val in subcommand[1].strip("()").split(",")].insert(1, [0.0])
                    case '-y':
                        new_points = [float(val)*alpha_l for val in subcommand[1].strip("()").split(",")].insert(1, [0.0])
                    case 'z':
                        new_points = [float(val)*alpha_l for val in subcommand[1].strip("()").split(",")] + [0.0]
                    case '-z':
                        new_points = [float(val)*alpha_l for val in subcommand[1].strip("()").split(",")] + [0.0]
                
                # Consider point outlier and exclude if too far from previous point
                # Also exclude if vector distance is 0
                # NOAH -- implement minimum threshold instead of 0? Consider multiple prev points if they exist?
                prev_points = np.asarray(ls.points).tolist() 
                vd1 = vector_distance(prev_points[-2], new_points)
                vd2 = vector_distance(prev_points[-1], prev_points[-2])
                
                if (vd1 != 0) and (vd1 < 30*vd2 or vd2 == 0):
                    add_this = o3d.utility.Vector3dVector(
                        np.array([new_points])
                    )
                    pcd.points.extend(add_this)
                    ls.points.extend(add_this)
                    ls.lines.extend(o3d.utility.Vector2iVector(np.array([[len(pcd.points) - 1, len(pcd.points)]])))
                    
                    # Update references and vis
                    ls_dict[ls_id] = ls
                    ls_dict[pcd_id] = pcd
                    vis.update_geometry(pcd)
                    vis.update_geometry(ls)

                    # Set max threshold of points to avoid unreasonable computational load
                    numPoints = len(np.asarray(pcd.points).tolist())
                    if numPoints >= 175:
                        rst_bit = 1
                    else:
                        main_window.update_dynamic_text(f"Sketching object point {numPoints}/175")

def clone_mesh(mesh):
    cloned_mesh = o3d.geometry.TriangleMesh()
    cloned_mesh.vertices = o3d.utility.Vector3dVector(np.array(mesh.vertices))
    cloned_mesh.triangles = o3d.utility.Vector3iVector(np.array(mesh.triangles))
    cloned_mesh.vertex_normals = o3d.utility.Vector3dVector(np.array(mesh.vertex_normals))
    cloned_mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(mesh.vertex_colors))
    # Need to copy all attributes (e.g. texture coordinates)
    return cloned_mesh

def handle_update_geo(subcommand, history, object_handle, vis, main_window, objects_dict, object_id, ls_dict):
    global extrusion_distance
    global prev_rotated
    global rst_bit
    
    alpha_m = 0.01  # Translation scaling factor
    alpha_r = 1     # Rotation scaling factor (in radians for Open3D)
    alpha_s = 100   # Scaling scaling factor
    alpha_e = 100   # Extrusion scaling factor
    direction = [1,0,0]

    match subcommand[1]:
        case "start":
            
            history["operation"] = subcommand[0]  # Operation type
            history["axis"] = subcommand[2] if len(subcommand) > 2 else None  # Axis, if applicable
            history["lastVal"] = subcommand[3] if len(subcommand) > 3 else None  # Starting value, if applicable
            
            if (subcommand[0] == "extrude" and prev_rotated):
                view_control = vis.get_view_control()
                cam_params = view_control.convert_to_pinhole_camera_parameters()
                current_extrinsic = cam_params.extrinsic
                updated_extrinsic = current_extrinsic.copy()
                updated_extrinsic[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

                smooth_transition(vis, view_control, updated_extrinsic)
                prev_rotated=False
                
        case "end":
            # Reset the history after operation ends
            if(history["operation"] == "extrude"):   
                object_handle.paint_uniform_color(selected_color)
                vis.update_geometry(object_handle)            
                history['total_extrusion_x'] = 0
                history['total_extrusion_y'] = 0

        case "position":
                match history["operation"]:
                    # Actually object translation in this case (reusing ML commands)
                    case "pan": 
                        main_window.update_dynamic_text("Translating object")
                        try:
                            
                            # Get delta of new and old xy coords
                            coords = subcommand[2].strip("()").split(",")
                            current_x = float(coords[0])
                            current_y = float(coords[1])
                            
                            old_coords = history["lastVal"].strip("()").split(",")
                            old_x = float(old_coords[0])
                            old_y = float(old_coords[1])
                            
                            delta_x = (current_x - old_x) * alpha_m
                            delta_y = (current_y - old_y) * alpha_m
                            
                            # Threshold for movement to be considered significant
                            threshold = 0.02

                            view_control = vis.get_view_control()
                            cam_params = view_control.convert_to_pinhole_camera_parameters()
                            rotation_matrix = np.linalg.inv(cam_params.extrinsic[:3, :3])

                            view_space_translation = np.array([delta_x if abs(delta_x) > threshold else 0, 
                                                            delta_y if abs(delta_y) > threshold else 0, 0])
                            world_space_translation = rotation_matrix.dot(view_space_translation)

                            # Only apply translation if there's significant movement
                            if np.linalg.norm(view_space_translation) > 0:
                                object_handle.translate(world_space_translation, relative=True)
                                objects_dict[object_id]['center'] = object_handle.get_center()
                                if 'original' in objects_dict[object_id]:
                                    objects_dict[object_id]['original'].translate(world_space_translation, relative=True)
                            
                            history["lastX"] = current_x
                            history["lastY"] = current_y
                        except Exception as e:
                            print(f"Error processing numerical values from command: {e}")
                            pass

                    case "rotate": 
                        try:
                            main_window.update_dynamic_text("Rotating object")
                            # Get delta and execute rotation
                            delta = float(subcommand[2]) - float(history["lastVal"])
                            rotate_object(object_handle, history["axis"], degrees=delta * alpha_r)
                            rotate_object( objects_dict[object_id]['original'], history["axis"], degrees=delta * alpha_r)

                        except Exception as e:
                            print(f"Error processing numerical values from command: {e}")
                            pass

                    case "zoom": 
                        try:
                            delta = float(subcommand[2]) - float(history["lastVal"])

                            selected_object_id = None
                            for object_id, obj_info in objects_dict.items():
                                # Skip the 'original' entry
                                if object_id == 'original':  
                                      continue
                                if obj_info.get('selected', False):
                                    new_scale = obj_info['scale'] + 100 * (delta / alpha_s)

                                    # Cap the new scale to be within the range [1, 200]
                                    if new_scale < 1:
                                        new_scale = 1
                                    elif new_scale > 200:
                                        new_scale = 200

                                    # Update the scale in the object info
                                    obj_info['scale'] = new_scale
                                    formatted_text = f"Scaling factor: {new_scale:.0f}%"
                                    main_window.update_dynamic_text(formatted_text)
                                    break  

                            min_size = 0.05                            
                            scale_object(object_handle, delta/alpha_s, min_size)
                            scale_object(objects_dict[object_id]['original'], delta/alpha_s, min_size)                

                        except Exception as e:
                            print(f"Error processing numerical values from command: {e}")
                            pass
                                
                    case "extrude":
                        try:
                            # Extrude the 2D shape if polygon count is not too high already
                            current_triangle_count = len(object_handle.triangles)
                            
                            if (current_triangle_count > 60000):
                                    main_window.update_dynamic_text("This object is too big to extrude")
                                    rst_bit = 1
                                    return

                            # Get delta of xy coordinates
                            coords = subcommand[2].strip("()").split(",")
                            current_x = float(coords[0])
                            current_y = float(coords[1])
                            
                            oldCoords = history["lastVal"].strip("()").split(",")
                            old_x = float(oldCoords[0])
                            old_y = float(oldCoords[1])
                            
                            delta_x = (current_x - old_x) / alpha_e
                            delta_y = (current_y - old_y) / alpha_e

                            object_handle.paint_uniform_color(closest_color)
                            
                            # Set defaults
                            factor = 1
                            maxLimit = 1
                            voxelFactor = 0.1
                            
                            # Retrieve the object type from the dictionary
                            object_type = objects_dict[object_id]['type']
                            
                            # Set extrusion parameter for the object type
                            match object_type:
                                case "cube":
                                    factor = 0.5
                                    maxLimit = 2.5
                                case "sphere":
                                    factor = 0.80
                                    maxLimit = 2
                                    voxelFactor = 0
                                case "triangle":
                                    factor = 0.5  
                                    maxLimit = 2
                                case "sketch":
                                    factor = 0.4  
                                    maxLimit = 1
                                    voxelFactor = 0.007
                                case _:
                                    factor = 1
                                    maxLimit = 1

                            extrusion_distance_x = delta_x
                            extrusion_distance_y = delta_y
                            history['last_extrusion_distance_x'] += delta_x
                            history['last_extrusion_distance_y'] -= delta_y

                            if 'total_extrusion_x' not in objects_dict[object_id]:
                                objects_dict[object_id]['total_extrusion_x'] = 0.0
                                
                            if 'total_extrusion_y' not in objects_dict[object_id]:
                                objects_dict[object_id]['total_extrusion_y'] = 0.0

                            # Calculate the new total extrusion considering this operation
                            new_total_extrusion_x = objects_dict[object_id]['total_extrusion_x'] + abs(extrusion_distance_x)
                            new_total_extrusion_y = objects_dict[object_id]['total_extrusion_y'] + abs(extrusion_distance_y)

                            main_window.update_dynamic_text("Extruding object")

                            if new_total_extrusion_x > maxLimit: #Double redundancy --theoretically shouldn't need these extra check as we check above if the object has too many triangles, but keeping it to be safe
                                pass

                            elif abs(history['last_extrusion_distance_x']) >= 0.20:
                                direction = [1,0,0]
                                if history['last_extrusion_distance_x'] < 0:
                                    direction = [-1,0,0]

                                objects_dict[object_id]['total_extrusion_x'] += 0.1

                                object_handle = custom_extrude(object_id, object_handle, direction, 0.1, vis, history, factor, voxelFactor)
                                
                            if new_total_extrusion_y > maxLimit: #Double redundancy --theoretically shouldn't need these extra check as we check above if the object has too many triangles, but keeping it to be safe
                                pass


                            elif abs(history['last_extrusion_distance_y']) >= 0.20:
                                objects_dict[object_id]['total_extrusion_y'] += 0.1
                                direction = [0,1,0]

                                if history['last_extrusion_distance_y'] < 0:
                                    direction = [0,-1,0]

                                object_handle = custom_extrude(object_id, object_handle, direction, 0.1, vis, history, factor, voxelFactor)

                        except Exception as e:
                            print(f"Error processing numerical values from command: {e}")
                         
                vis.update_geometry(object_handle)
                history["lastVal"] = subcommand[2]



def custom_extrude(object_id, object_handle, direction, distance, vis, history, factor, voxelFactor):
    vis.remove_geometry(object_handle, reset_bounding_box=False)

    #Save the original state if not saved yet
    if 'original' not in objects_dict[object_id]:
        objects_dict[object_id]['original'] = clone_mesh(object_handle)

    # Get vertices and faces from the original mesh
    vertices = np.asarray(object_handle.vertices)
    faces = np.asarray(object_handle.triangles)
    
    # Calculate new vertices by adding the extrusion direction and distance to the original vertices
    new_vertices = vertices + np.array(direction) * distance
    
    # Create faces for the sides of the extrusion. This assumes a closed, watertight mesh
    side_faces = []
    for edge in object_handle.get_non_manifold_edges():
        v1, v2 = edge
        # Create two triangles for each edge to form a quadrilateral side face
        new_face_1 = [v1, v2, v2 + len(vertices)]
        new_face_2 = [v1, v2 + len(vertices), v1 + len(vertices)]
        side_faces.append(new_face_1)
        side_faces.append(new_face_2)
    
    # Combine original and new vertices and faces
    all_vertices = np.vstack((vertices, new_vertices))
    all_faces = np.vstack((faces, faces + len(vertices)))
    
    # Check if side_faces is not empty before concatenating
    if side_faces:
        all_faces = np.vstack((all_faces, side_faces))

    # Create a new mesh with the combined data
    extruded_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(all_vertices),
        triangles=o3d.utility.Vector3iVector(all_faces)
    )

    current_triangle_count = len(extruded_mesh.triangles)
    
    # Recompute normals for the new mesh
    simplified_mesh = extruded_mesh.simplify_quadric_decimation(target_number_of_triangles=round(current_triangle_count* factor))
    if voxelFactor > 0:
         simplified_mesh = extruded_mesh.simplify_vertex_clustering(voxel_size = voxelFactor)
    simplified_mesh.compute_vertex_normals()
    
    history['last_extrusion_distance_x'] = 0.0
    history['last_extrusion_distance_y'] = 0.0
    
    # Update the visualizer and object dictionary
    vis.add_geometry(simplified_mesh, reset_bounding_box=False)
    objects_dict[object_id]['object'] =  simplified_mesh
    return extruded_mesh


def handle_selection(objects_dict, vis, main_window):
    for object_id, obj_info in objects_dict.items():

        # Check if the 'highlighted' key exists and is True
        if obj_info.get('highlighted', False):  
            obj_info['selected'] = True
            obj = obj_info['object']  
            obj.paint_uniform_color(selected_color)
            
            vis.update_geometry(obj)
            main_window.update_dynamic_text(f"Object selected!")
            return 1

    return 0

def handle_deselection(objects_dict, vis, main_window):
    for object_id, obj_info in objects_dict.items():

        # Check if the 'selected' key exists and is True
        if obj_info.get('selected', False):  
            obj_info['selected'] = False
            obj_info['highlighted'] = False
            obj = obj_info['object']
            obj.paint_uniform_color([0.5, 0.5, 0.5])
            
            vis.update_geometry(obj)
            main_window.update_dynamic_text(f"Object {object_id} deselected")
            main_window.update_progress(0)
            
    return 0

def handle_commands(clientSocket, vis, view_control, camera_parameters, geometry_dir, history, objects_dict, counters, main_window):
    try:
        # Attempt to receive data, but don't block indefinitely
        command = get_tcp_data(clientSocket, len(ls_dict))

        if command:
            # Parse and handle the command
            parse_command(command, view_control, camera_parameters, vis, geometry_dir, history, objects_dict, counters, main_window)
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
    # Set up UI window
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
    serverSocket = start_server()
    clientSocket = make_connection(serverSocket);

    # Set up Open3D window and camera
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
    
    # Create default mesh
    print("Testing mesh in Open3D...")
    mesh = o3d.geometry.TriangleMesh.create_box(width=0.2, height=0.4, depth=0.2)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.5, 0.5, 0.5]) 

    # Create visualizer and window
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Open3D", width=width, height=height, left=50, top=50, visible=True
    )
    vis.add_geometry(mesh)
    vis.get_render_option().background_color = np.array([0.25, 0.25, 0.25])
    
    # Add the grid to the visualizer
    grid = create_grid(size=15, n=20, plane='xz', color=[0.5, 0.5, 0.5])
    vis.add_geometry(grid)
    
    # Set up camera draw distance
    camera = vis.get_view_control()
    camera.set_constant_z_far(4500)

    # Object tracking dictionaries
    objects_dict['object_1'] = {'object': mesh, 'center': mesh.get_center(), 'highlighted' : False, 'selected' : False,  'scale' : 100, 'type': "cube"}
    ls_dict = {}
    counters = {"ls":0, "pcd":0, "meshes":0}

    # View control
    view_control = vis.get_view_control()

    # Initialize required dictionaries and parameters
    geometry_dir = {"counters": {"pcd": 0, "ls": 0, "mesh": 0}} # NOAH -- REMOVE
    history = {"operation": "", "axis": "", "lastVal": "", 'last_extrusion_distance_x': 0.0,'last_extrusion_distance_y': 0.0, 'total_extrusion': 0.0}

    main_window = MainWindow(vis)
    main_window.show()
    snap_isometric(vis, view_control)

    # Setup a QTimer to periodically check for new commands
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: handle_commands(clientSocket, vis, view_control, camera_parameters, geometry_dir, history, objects_dict, counters, main_window))
    timer.start(13)  # Check every 100 milliseconds
    

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
