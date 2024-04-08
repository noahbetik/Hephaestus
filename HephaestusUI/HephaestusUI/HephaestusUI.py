from csv import list_dialects
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

# ----------------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------------------------


rst_bit = 0

objects_dict = {}
ls_dict = {}
curr_highlighted = False
prevRotated = True
prevAdded = False
prevSnapped = False
marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
previous_look_at_point = None
zoomFactor = 0.25
extrusion_distance = 0
deleteCount = 0
selected_pkt = 0
curr_pkt = 0


regularColor = [0.5, 0.5, 0.5]
closestColor = [0.0, 0.482, 1.0]
selectedColor = [0.157, 0.655, 0.271]
backgroundColor = [0.11, 0.11, 0.11]
gridColor = [0.29, 0.29, 0.29]

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
        #Update the progress bar with the new value
        self.progress_bar.setValue(value)

    def toggle_full_screen(self):
        if self.isFullScreen():
            self.showNormal()  # If the window is in full-screen mode, exit full-screen
            self.fullscreen_button.setText('Full Screen')  # Update button text
        else:
            self.showFullScreen()  # Enter full-screen mode
            self.fullscreen_button.setText('Exit Full Screen')  # Update button text


    def on_action_button_clicked(self):
        global objects_dict, ls_dict, snapCount, curr_highlighted, prevRotated ,prevAdded, prevSnapped , extrusion_distance, deleteCount, rst_bit
        # This method will be called when the button is clicked
        print("Reset button pressed!")
        curr_highlighted = False
        prevRotated = True
        prevAdded = False
        prevSnapped = False   
        extrusion_distance = 0
        deleteCount = 0
        rst_bit = 1


        for obj in objects_dict.values():
            self.vis.remove_geometry(obj['object'], reset_bounding_box=False)

        objects_dict = {}

        for k, v in ls_dict.items():
            self.vis.remove_geometry(ls_dict[k], reset_bounding_box=False) # B===D

        ls_dict = {}

        snap_isometric(self.vis, self.vis.get_view_control())
        self.update_progress(0)
        
        

    def update_dynamic_text(self, new_text):
        self.dynamic_text_widget.setText(new_text)

    def update_mode_text(self, new_text):
        self.mode_text_widget.setText(f"Mode: {new_text}")


def create_grid(size=10, n=10, plane='xz', color=regularColor):
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

    #predefined extrinsic set for the isometric view
    target_extrinsic = np.array([
        [ 8.68542871e-01, -1.11506045e-03,  4.95612791e-01,  1.71200000e-01],
        [-2.08451221e-01, -9.08069551e-01,  3.63259933e-01,  7.36100000e-02],
        [ 4.49645828e-01, -4.18817916e-01, -7.88929771e-01,  1.99985850e+00],  # Your comment here
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
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


def smooth_transition(vis, view_control, target_extrinsic, steps=85):
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



def move_camera_v2(view_control, direction, amount=1.5):
    cam_params = view_control.convert_to_pinhole_camera_parameters()
    extrinsic = np.array(cam_params.extrinsic)

    if direction == "x":
        extrinsic[2, 3] -= amount
    elif direction == "y":
        extrinsic[0, 3] -= amount
    elif direction == "z":
        extrinsic[1, 3] -= amount

    cam_params.extrinsic = extrinsic
    view_control.convert_from_pinhole_camera_parameters(cam_params, True)
    
def move_camera_v3(view_control, vals, threshold=0.015):
    """
    Moves the camera based on the provided values, ignoring movements that are below a specified threshold.
    
    Parameters:
    - view_control: The view control object to manipulate the camera's extrinsic parameters.
    - vals: A tuple or list with two elements indicating the amount of movement in the y and z axes, respectively.
    - threshold: The minimum movement required to apply the camera movement. Movements below this threshold are ignored.
    """
    cam_params = view_control.convert_to_pinhole_camera_parameters()
    extrinsic = np.array(cam_params.extrinsic)

    # Initialize a variable to track if any significant movement occurred
    significant_movement = False

    # Check and apply movement for the y axis if it exceeds the threshold
    if abs(vals[0]) > threshold:
        extrinsic[0, 3] += vals[0]
        significant_movement = True
    else:
        print("Y-axis movement below threshold, ignoring.")

    # Check and apply movement for the z axis if it exceeds the threshold
    if abs(vals[1]) > threshold:
        extrinsic[1, 3] += vals[1]
        significant_movement = True
    else:
        print("Z-axis movement below threshold, ignoring.")

    # If any significant movement occurred, update the camera parameters
    if significant_movement:
        cam_params.extrinsic = extrinsic
        view_control.convert_from_pinhole_camera_parameters(cam_params, True)
    else:
        print("No significant movement detected.")



def rotate_camera(view_control, axis, degrees=5):
    global zoomFactor
    
    angle = np.radians(degrees)
    if abs(angle) < 0.01:
        return

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
    threshold = 0.15  # tune
    if (
        abs(endPoint[0] - startPoint[0]) < threshold
        and abs(endPoint[1] - startPoint[1]) < threshold
    ):
        return startPoint
    else:
        return endPoint

def smartConnectBool(endPoint, startPoint):
    threshold = 0.15  # tune
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

def getTCPData(sock, sketch):
    global tcp_command_buffer
    global selected_pkt, rst_bit, curr_pkt
    try:
        # Temporarily set a non-blocking mode to check for new data
        
        #send reset
        if rst_bit == 1: 
            sock.sendall("RST".encode("ascii"))
            print("sent RST bit via TCP! rst = ", rst_bit)
            rst_bit = 0
        
        if selected_pkt == 1 and curr_pkt == 0:
            sock.sendall("SEL".encode("ascii"))
            print("sent SEL bit via TCP! pkt = ", selected_pkt)
            curr_pkt = 1
        if selected_pkt == 0 and curr_pkt == 1:
            sock.sendall("DESEL".encode("ascii"))
            print("sent DESEL bit via TCP!  pkt= ", selected_pkt)
            curr_pkt = 0

        
        sock.setblocking(False)
        
        packet = "ACK"
        sock.sendall(packet.encode("ascii"))
        print("sent back ", packet)
        data = sock.recv(1024)  # Attempt to read more data
      


        if data:
        # Process received data
            print(f"Received: {data.decode('ascii').strip()}")

            # Send acknowledgment back
           # sock.sendall(packet.encode("ascii"))
            #print("sent back ", packet)
        #sock.setblocking(True)  # Revert to the original blocking mode

        # Decode and append to buffer
        tcp_command_buffer += data.decode('ascii')

        # Process if delimiter is found
        if '\n' in tcp_command_buffer:
            command, tcp_command_buffer = tcp_command_buffer.split('\n', 1)
            command = command.strip("$")  # Strip any '$' characters as in your original processing
           # print("Received: ", command)
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


def sketchExtrude(counters, vis):
    
    global ls_dict
    global objects_dict

    print("SKETCH EXTRUDE!!!")

    ls_id = "ls" + str(counters["ls"] -1)
    pcd_id = "pcd" + str(counters["pcd"] -1)

    print(ls_dict)
    
    ls = ls_dict[ls_id]
    pcd = ls_dict[pcd_id]

    #ps = [[0,0,0], [1,0,0], [1.5,1,0], [1,2,0], [0,2,0], [-0.5,1,0]] # --> original sketch
    #ps2 = [[0,0,1], [1,0,1], [1.5,1,1], [1,2,1], [0,2,1], [-0.5,1,1]] # --> desired opposite prism face

    scaleFactor = 5
    stackFactor = 5
    stackHeight = 0.1
    stepSize = stackHeight / stackFactor

    points = np.asarray(pcd.points).tolist()

    print("interpolating sketch")

    for p in range(len(points) - 1): # increase point density of original sketch
        points = points + linear_interpolate_3d(points[p], points[p+1], scaleFactor)
    points = points + linear_interpolate_3d(points[-1], points[0], scaleFactor)

    stacked = []

    print("creating stack")

    for p in points:
        for i in np.arange(stepSize, 0.1, stepSize):
            stacked.append([p[0], p[1], i])

    points2 = [[x, y, z+stackHeight] for x, y, z in points] # create opposite face
    
    #pcd.points.extend(o3d.utility.Vector3dVector(np.array(stacked)))

    print("creating scaled faces")

    totalScaled = []

    for i in range (0, scaleFactor):
        print(f"scaling faces down to {i/scaleFactor}")    
        scaled1 = scale_polygon_2d(np.array(points), i/scaleFactor)
        scaled2 = scale_polygon_2d(np.array(points2), i/scaleFactor)
        
        totalScaled = totalScaled + scaled1 + scaled2

    points = points + points2 + stacked + totalScaled
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    vis.update_geometry(pcd)

    print("all points done")
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
    

    # Visualize the point cloud
    vis.remove_geometry(ls)
    vis.remove_geometry(pcd)
    ls_dict = {}
    #vis.add_geometry(mesh)
    addGeometry(vis, mesh, objects_dict, "sketch", False)

    

# ----------------------------------------------------------------------------------------
# PARSE COMMAND
# ----------------------------------------------------------------------------------------


def parseCommand(
    command, view_control, camera_parameters, vis, geometry_dir, history, objects_dict, counters, main_window
):
    # FORMAT
    # [command] [subcommand]

    global prevRotated
    global deleteCount
    global prevAdded
    global selected_pkt
    info = command.split(" ")
    print(info)
    objectHandle = ""    
    object_id = ""

    # Check for selected objects in objects_dict and update objectHandle to point to the mesh if selected
    for id, obj_info in objects_dict.items():
        if obj_info.get('selected', False):  # Check if the 'selected' key exists and is True
            objectHandle = obj_info['object']  # Now, objectHandle directly references the mesh object
            object_id = id
           #print(f"Selected object: {id}")
            break  # Assume only one object can be selected at a time; break after finding the first selected object


    match info[0]:
        case "motion":
            
            #print("**************************** ", info[1:][0])

            if objectHandle == "" and info[1:][0] != "extrude":
                main_window.update_mode_text("Camera")
                handleCam(info[1:], view_control, history, vis, main_window)
                return ""

            elif objectHandle:
                # if (prevRotated) : snap_to_closest_plane(vis, view_control)
                # prevRotated = False
                main_window.update_mode_text("Object")
                handleUpdateGeo(info[1:], history, objectHandle, vis, main_window, objects_dict, object_id, ls_dict)
                objectHandle.paint_uniform_color(selectedColor)  #back to green 
                return ""
        case "select":
            if len(ls_dict) == 2:
                print("wehehehehe")
                #force deselect in case something is selected
                camera_parameters = vis.get_view_control().convert_to_pinhole_camera_parameters()
                handleDeselection(objects_dict, vis, main_window)
                
                sketchExtrude(counters, vis)

                smooth_transition(vis, view_control, np.array([
                [ 0.99994158, -0.00229353,  0.0105631 , -1.22174849],
                [-0.00238567, -0.99995914,  0.00871879,  0.39801206],
                [ 0.01054267, -0.00874348, -0.9999062 ,  1.79901982],  # Your comment here
                [ 0.        ,  0.        ,  0.        ,  1.        ]
            ]), steps = 25)
            else:
                prevAdded = False
                deleteCount = 0
                main_window.update_mode_text("Object")
                selected_pkt = handleSelection(objects_dict, vis, main_window)  # Assume this function handles object selection
        case "deselect":
            deleteCount = 0
            selected_pkt  = 0

            main_window.update_mode_text("Camera")
            selected_pkt = handleDeselection(objects_dict, vis, main_window)  # Assume this function handles object selection
        case "create":
            deleteCount = 0
            if not prevAdded:
                handleNewGeo(info[1:], view_control, camera_parameters, vis, objects_dict, counters, main_window)
                handleDeselection(objects_dict, vis, main_window)
                #prevAdded = True
            return ""
        case "update":
            deleteCount = 0


            # if objectHandle:
            #     if (prevRotated) : snap_to_closest_plane(vis, view_control)
            #     prevRotated = False
            handleUpdateGeo(info[1:], history, objectHandle, vis, main_window, objects_dict, object_id, ls_dict)
            

        case "snap":
            if info[1:][0] == "iso":
                snap_isometric(vis, view_control)
            elif info[1:][0] == "home":
                snap_to_closest_plane(vis, view_control)
 
            
        case "delete":
            deleteCount+=1
            if (deleteCount>5 and objectHandle):
                removeGeometry(vis, objectHandle, object_id)
                deleteCount = 0

        
            try:
                if 'original' in objects_dict[object_id] and object_id:
                    history['total_extrusion_x'] = 0
                    history['total_extrusion_y'] = 0

                    vis.remove_geometry(objectHandle, reset_bounding_box=False)
                    objectHandle = clone_mesh(objects_dict[object_id]['original'])
                    objectHandle.compute_vertex_normals()
                    objectHandle.paint_uniform_color(selectedColor)  #back to green 
                    vis.add_geometry(objectHandle, reset_bounding_box=False)
                    vis.update_geometry(objectHandle)


                    objects_dict[object_id]['object'] = objectHandle  # Update the current object with the original
                    history['last_extrusion_distance_x'] = 0.0
                    history['last_extrusion_distance_y'] = 0.0

                    objects_dict[object_id]['total_extrusion_x'] = 0.0
                    objects_dict[object_id]['total_extrusion_y'] = 0.0
                    objects_dict[object_id]['selected'] = True
            except KeyError:
                pass
        case "lock-in":
            main_window.update_progress(int(info[1]))
        case _:
            history["lastVal"] = info[1:][2]

            

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
            obj.paint_uniform_color(closestColor)  # Light green for highlighted object
            info['highlighted'] = True
        else:
            # Unhighlight all other objects by resetting their color
            obj.paint_uniform_color(regularColor)  # Default color
            info['highlighted'] = False
        
        # Update the geometry to reflect changes
        vis.update_geometry(obj)

    # You might need to call this if the visualizer doesn't automatically refresh
    vis.update_renderer()


def removeGeometry(vis, obj, id):
    print("deleting object ", id)

    vis.remove_geometry(obj, reset_bounding_box = False)
    del objects_dict[id]
    print("deleted object ", id)
    return

def addGeometry(vis, obj, objects_dict, objType, wasSpawned):
    # Generate a unique object ID based on the current number of items in objects_dict
    object_id = f"object_{len(objects_dict) + 1}"
    
    # Compute the center of the object for translation to the origin
    center = obj.get_center()

    # Update the object's color
    obj.paint_uniform_color([0.5, 0.5, 0.5])  # Reset the object color to grey
    
    if wasSpawned:
        print("was spawned")
        translation_vector = -center  # Vector to move the object's center to the origin
    
        # Translate the object to the origin
        obj.translate(translation_vector, relative=False)
        obj.translate(np.array([0, 0.2, 0]))
        # Add the object to the visualizer
        vis.add_geometry(obj, reset_bounding_box=False)
    else:
        print("was sketched")
        # Add the object to the visualizer
        vis.add_geometry(obj, reset_bounding_box = False)
    


    
    # Add the new object to objects_dict with its properties
    objects_dict[object_id] = {
        'object': obj, 
        'center': center,  # Object is now at the origin
        'highlighted': False, 
        'selected': False, 
        'scale': 100,
        'type' : objType
    }
    
              

def scale_object(objectHandle, delta, min_size=0.01, max_size=1.5):
    # Intended scale factor based on the delta
    scaleFactor = 1 + delta
    
    # Get the object's bounding box
    bbox = objectHandle.get_axis_aligned_bounding_box()
    extents = bbox.get_extent()  # This gives you the width, height, and depth of the bounding box
    
    # Calculate the new extents after scaling
    new_extents = extents * scaleFactor
    
    # Check if any of the new extents fall below the minimum size
    if np.any(new_extents < min_size):
        print("Adjusting scale to prevent object from getting too small.")
        required_scale_factors_for_min = min_size / extents
        scaleFactor = max(required_scale_factors_for_min.max(), scaleFactor)

    # Check if any of the new extents exceed the maximum size
    elif np.any(new_extents > max_size):
        print("Adjusting scale to prevent object from getting too large.")
        required_scale_factors_for_max = max_size / extents
        scaleFactor = min(required_scale_factors_for_max.min(), scaleFactor)

    # If the final scale factor is significantly different from 1 + delta, we have made an adjustment
    print(f"Adjusted scale factor: {scaleFactor:.2f}")
    
    # Compute the object's center for uniform scaling about its center
    center = bbox.get_center()
    
    # Apply the scaling
    objectHandle.scale(scaleFactor, center)



def rotate_object(objectHandle, axis, degrees=5):
    # Calculate the rotation angle in radians
    angle = np.radians(degrees)
    if abs(angle) < 0.02:
        return
    
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
    elif axis == "z":
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0, 0],
            [np.sin(angle), np.cos(angle), 0, 0],
            [0, 0, 1, 0],
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
    
    
def handleCam(subcommand, view_control, history, vis, main_window):
    
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
           # print(history)
        case "end":
            print("ending motion")
            main_window.update_dynamic_text("Waiting for command...")
            history["operation"] = ""
            history["axis"] = ""
            history["lastVal"] = ""
            return
        case "position":
            match history["operation"]:
                case "pan":
                    print("camera pan update")
                    main_window.update_dynamic_text("Panning camera")
                    splitCoords = subcommand[2].strip("()").split(",")
                    oldCoords = history["lastVal"].strip("()").split(",")
                    deltaY = (float(splitCoords[0]) - float(oldCoords[0])) * alphaM
                    deltaZ = (float(splitCoords[1]) - float(oldCoords[1])) * alphaM
                    move_camera_v3(view_control, [deltaY, deltaZ])

                    highlight_objects_near_camera(vis, view_control, objects_dict)
                    
                case "rotate":
                    print("current axis, ", history["axis"])
                    main_window.update_dynamic_text("Rotating camera")
                    prevRotated = True
                    print("camera rotate update")
                    delta = float(subcommand[2]) - float(history["lastVal"])
                    rotate_camera(view_control, history["axis"], degrees=delta * alphaR)
                    highlight_objects_near_camera(vis, view_control, objects_dict)


                case "zoom":
                    print("camera zoom update")
                    main_window.update_dynamic_text("Zooming camera")
                    delta = float(subcommand[2]) - float(history["lastVal"])
                    move_camera_v2(view_control, "x", delta * alphaZ)
                    highlight_objects_near_camera(vis, view_control, objects_dict)

            history["lastVal"] = subcommand[2]
        case _:
            print("INVALID COMMAND")


def handleNewGeo(subcommand, view_control, camera_parameters, vis, objects_dict, counters, main_window):
    global view_axis
    global prevAdded
    global ls_dict

    alphaL = 0.002 # line scaling factor (maybe better way to do this)
    #print("***********subcommand[0] ", subcommand[0])
    

    match subcommand[0]:
        case "cube":
            print("Creating new box at origin")
            # Store the current view matrix
            current_view_matrix = view_control.convert_to_pinhole_camera_parameters().extrinsic
            # Create and add the cube
            new_box = o3d.geometry.TriangleMesh.create_box(width=0.2, height=0.2, depth=0.2)
            new_box.compute_vertex_normals()
            addGeometry(vis, new_box, objects_dict, "cube", True)
            prevAdded = True

        case "sphere":
            print("Creating new sphere at origin")
            # Store the current view matrix
            current_view_matrix = view_control.convert_to_pinhole_camera_parameters().extrinsic
            # Create and add the sphere
            new_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
            new_sphere.compute_vertex_normals()
            addGeometry(vis, new_sphere, objects_dict, "sphere", True)
            prevAdded = True


        case "triangle":
            print("Creating new triangle at origin")
            # Store the current view matrix
            current_view_matrix = view_control.convert_to_pinhole_camera_parameters().extrinsic

            # Manually define a larger triangle mesh with depth
            scale_factor = 2  # Increase this factor to scale up the size of the triangle
            depth = 0.1  # The depth of the triangle in the z-axis
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
            addGeometry(vis, new_triangle, objects_dict, "triangle", True)
            prevAdded = True


        case "line":  # line handling not fully implemented yet
            main_window.update_dynamic_text("Drawing new line")

            # Store the current view matrix
            current_view_matrix = (
                view_control.convert_to_pinhole_camera_parameters().extrinsic
            )
            #print("**********SUBCOMMAND IS ",subcommand[1])

            if subcommand[1] == "start":
          

                smooth_transition(vis, view_control, np.array([
                [ 9.99986618e-01, -1.10921734e-03,  5.05311841e-03, -1.28562713e+00],
                [-1.13032407e-03, -9.99990642e-01,  4.17603074e-03,  4.10222709e-01],
                [ 5.04843899e-03, -4.18168652e-03, -9.99978513e-01,  2.46971612e+00],  # Your comment here
                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
            ]))

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
                main_window.update_dynamic_text("Waiting for command...")

                ls_id = "ls" + str(counters["ls"] - 1)
                pcd_id = "pcd" + str(counters["pcd"] - 1)
                ls = ls_dict[ls_id]
                pcd = ls_dict[pcd_id]
                print("Ending line")
                # threshold for connecting closed-loop geometry
                all_points = np.asarray(pcd.points).tolist()
                #print(pcd.points)
               # print(all_points)
                if smartConnectBool(all_points[-1], all_points[0]):
                    print("CONNECTY WECTY")
                    ls.lines.extend(o3d.utility.Vector2iVector(np.array([[len(pcd.points), 0]])))
                else:
                    print("NO CONNECTY WECTY")
                    
                ls_dict[ls_id] = ls
                ls_dict[pcd_id] = pcd

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
                
                if (vd1 != 0) and (vd1 < 30*vd2 or vd2 == 0):
                    add_this = o3d.utility.Vector3dVector(
                        np.array([new_points])
                    )
                    pcd.points.extend(add_this)  # get proper references for this
                    ls.points.extend(add_this)  # get proper references for this
                    ls.lines.extend(o3d.utility.Vector2iVector(np.array([[len(pcd.points) - 1, len(pcd.points)]])))
                    
                    ls_dict[ls_id] = ls
                    ls_dict[pcd_id] = pcd

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

def handleUpdateGeo(subcommand, history, objectHandle, vis, main_window, objects_dict, object_id, ls_dict):
    alphaM = 0.01  # Translation scaling factor
    alphaR = 1  # Rotation scaling factor (in radians for Open3D)
    alphaS = 100  # Scaling scaling factor
    alphaE = 100 #extrusion scaling factor
    direction = [1,0,0]

    
    
    global extrusion_distance
    global prevRotated
    global rst_bit

    match subcommand[1]:
        case "start":
            
            history["operation"] = subcommand[0]  # Operation type
            history["axis"] = subcommand[2] if len(subcommand) > 2 else None  # Axis, if applicable
            history["lastVal"] = subcommand[3] if len(subcommand) > 3 else None  # Starting value, if applicable
            
            
            #print("subcommand is, ",subcommand[0], " and prevRotated is ",prevRotated)
            if (subcommand[0] == "extrude" and prevRotated):
                view_control = vis.get_view_control()
                cam_params = view_control.convert_to_pinhole_camera_parameters()
                current_extrinsic = cam_params.extrinsic

                
                updated_extrinsic = current_extrinsic.copy()
                updated_extrinsic[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

                smooth_transition(vis, view_control, updated_extrinsic)
                prevRotated=False
        case "end":
            #print("Ending motion")
            # Reset the history after operation ends
            main_window.update_dynamic_text("Waiting for command...")
            
            if(history["operation"] == "extrude"):   
                objectHandle.paint_uniform_color(selectedColor) #back to selected
                vis.update_geometry(objectHandle)            
                history['total_extrusion_x'] = 0
                history['total_extrusion_y'] = 0


        case "position":
                match history["operation"]:
                    case "pan": #not actually pan, but object translation
                        #print("Updating position or transformation")
                        main_window.update_dynamic_text("Translating object")

                        try:
                            coords = subcommand[2].strip("()").split(",")
                            currentX = float(coords[0])
                            currentY = float(coords[1])
                            
                            oldCoords = history["lastVal"].strip("()").split(",")
                            
                            oldX = float(oldCoords[0])
                            oldY = float(oldCoords[1])
                            
                            deltaX = (currentX - oldX) * alphaM
                            deltaY = (currentY - oldY) * alphaM
                            # Define a threshold for movement to be considered significant.
                            threshold = 0.02  # Adjust this value based on your requirements

                            view_control = vis.get_view_control()
                            cam_params = view_control.convert_to_pinhole_camera_parameters()
                            rotation_matrix = np.linalg.inv(cam_params.extrinsic[:3, :3])

                            view_space_translation = np.array([deltaX if abs(deltaX) > threshold else 0, 
                                                            deltaY if abs(deltaY) > threshold else 0, 0])
                            world_space_translation = rotation_matrix.dot(view_space_translation)

                            # Only apply translation if there's significant movement
                            if np.linalg.norm(view_space_translation) > 0:
                                objectHandle.translate(world_space_translation, relative=True)
                                objects_dict[object_id]['center'] = objectHandle.get_center()
                                if 'original' in objects_dict[object_id]:
                                    objects_dict[object_id]['original'].translate(world_space_translation, relative=True)
      
                            #print("Translating object by", world_space_translation)
                            
                            history["lastX"] = currentX
                            history["lastY"] = currentY
                        except Exception as e:
                            print(f"Error processing numerical values from command: {e}")
                            pass

                           
                    case "rotate": 
                        
                        try:
                            #print("object rotation")
                            main_window.update_dynamic_text("Rotating object")
                            delta = float(subcommand[2]) - float(history["lastVal"])
                            rotate_object(objectHandle, history["axis"], degrees=delta * alphaR)
                            rotate_object( objects_dict[object_id]['original'], history["axis"], degrees=delta * alphaR)
                           

                            #print("current degrees ", delta * alphaR)
                        except Exception as e:
                            print(f"Error processing numerical values from command: {e}")
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
                                    new_scale = obj_info['scale'] + 100 * (delta / alphaS)

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



                            else:
                                #print("No object is currently selected.")
                                pass
                            min_size = 0.05                            
                            scale_object(objectHandle, delta/alphaS, min_size)
                            scale_object(objects_dict[object_id]['original'], delta/alphaS, min_size)                

                        except Exception as e:
                            print(f"Error processing numerical values from command: {e}")
                            pass
                                
                    case "extrude":
                        try:
                            
                            # if object_id != "" and len(ls_dict) == 1:
                            #     #extrude the 2d shape
                            current_triangle_count = len(objectHandle.triangles)
                            
                            if (current_triangle_count > 60000):
                                    main_window.update_dynamic_text("This object is too big to extrude")
                                    rst_bit = 1
                                    return
                                

                            coords = subcommand[2].strip("()").split(",")
                            currentX = float(coords[0])
                            currentY = float(coords[1])
                            
                            oldCoords = history["lastVal"].strip("()").split(",")
                            
                            oldX = float(oldCoords[0])
                            oldY = float(oldCoords[1])
                            
                            deltaX = (currentX - oldX) / alphaE
                            deltaY = (currentY - oldY) / alphaE

                            objectHandle.paint_uniform_color(closestColor)  # Set to light blue
                            
                            print("delta x is ",deltaX," delta y is ", deltaY)
                            
                            
                            factor = 1  # Default factor for scaling or other operations
                            maxLimit = 1
                            voxelFactor = 0.1
                            object_type = objects_dict[object_id]['type']  # Retrieve the object type from the dictionary

                            match object_type:
                                case "cube":
                                    factor = 0.5
                                    maxLimit = 1.5
                                case "sphere":
                                    factor = 0.80
                                    maxLimit = 0.8
                                    voxelFactor = 0
                                case "triangle":
                                    factor = 0.5  
                                    maxLimit = 1.5
                                case "sketch":
                                    factor = 0.4  
                                    maxLimit = 1
                                    voxelFactor = 0.007
                                case _:  # Default case if none of the above match
                                    factor = 1
                                    maxLimit = 1
                            print("factor set to ", factor, " max limie set to ",maxLimit)
                                                    
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
                            main_window.update_dynamic_text("Extruding object")
                            
                            print("----------------------------- total x is ",  objects_dict[object_id]['total_extrusion_x'])

                            if new_total_extrusion_x > maxLimit:
                                print("Maximum extrusion limit reached. No further extrusion will be performed.")
                                main_window.update_dynamic_text("Maximum extrusion limit in x direction reached. No further extrusion will be performed.")
                                pass

                            elif abs(history['last_extrusion_distance_x']) >= 0.20:##
                                direction = [1,0,0]
                                if history['last_extrusion_distance_x'] < 0:
                                    direction = [-1,0,0]

                                objects_dict[object_id]['total_extrusion_x'] += 0.1
                                print("extruding by 0.2 in x")

                                objectHandle = custom_extrude(object_id, objectHandle, direction, 0.1, vis, history, factor, voxelFactor)
                                
                            if new_total_extrusion_y > maxLimit:
                                print("Maximum extrusion limit reached. No further extrusion will be performed.")
                                main_window.update_dynamic_text("Maximum extrusion limit in y direction reached. No further extrusion will be performed.")
                                pass


                            elif abs(history['last_extrusion_distance_y']) >= 0.20:##
                                objects_dict[object_id]['total_extrusion_y'] += 0.1
                                direction = [0,1,0]

                                if history['last_extrusion_distance_y'] < 0:
                                    direction = [0,-1,0]
                                    
                                    

                                objectHandle = custom_extrude(object_id, objectHandle, direction, 0.1, vis, history, factor, voxelFactor)

                                #delta = deltaY
                            print("extrusion direction set to ", direction)
                                        
                                
                        except Exception as e:
                            print(f"Error processing numerical values from command: {e}")
                         
                         
                vis.update_geometry(objectHandle)
                history["lastVal"] = subcommand[2]



def custom_extrude(object_id, objectHandle, direction, distance, vis, history, factor, voxelFactor):
    vis.remove_geometry(objectHandle, reset_bounding_box=False)

    #Save the original state if not saved yet
    if 'original' not in objects_dict[object_id]:
        objects_dict[object_id]['original'] = clone_mesh(objectHandle)

    # Get vertices and faces from the original mesh
    vertices = np.asarray(objectHandle.vertices)
    faces = np.asarray(objectHandle.triangles)
    
    # Calculate new vertices by adding the extrusion direction and distance to the original vertices
    new_vertices = vertices + np.array(direction) * distance
    
    # Create faces for the sides of the extrusion. This assumes a closed, watertight mesh
    side_faces = []
    for edge in objectHandle.get_non_manifold_edges():
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
    
    print("--------------------------------------------------------Current triangle count ",current_triangle_count)
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


def extrude(object_id, objectHandle, objects_dict, vis, history, direction):
    vis.remove_geometry(objectHandle, reset_bounding_box=False)

    # Save the original state if not saved yet
    if 'original' not in objects_dict[object_id]:
        objects_dict[object_id]['original'] = clone_mesh(objectHandle)

    direction_tensor = o3d.core.Tensor(direction, dtype=o3d.core.Dtype.Float32)

    # Perform the extrusion
    mesh_extrusion = o3d.t.geometry.TriangleMesh.from_legacy(objectHandle)
    extruded_shape = mesh_extrusion.extrude_linear(direction_tensor, scale=0.2)

    # Simplify the extruded shape before converting to legacy
    # Note: You might need to convert the tensor mesh back to a legacy mesh for simplification
    # if the tensor-based mesh does not directly support simplification.
    
    simplified_extruded_shape = extruded_shape.to_legacy()#.simplify_quadric_decimation(target_number_of_triangles=750000)

    simplified_extruded_shape.compute_vertex_normals()
    simplified_extruded_shape.paint_uniform_color(closestColor)  # Set to light blue

    # Update the visualizer and object dictionary
    vis.add_geometry(simplified_extruded_shape, reset_bounding_box=False)
    objects_dict[object_id]['object'] = simplified_extruded_shape

    history['last_extrusion_distance_x'] = 0.0
    history['last_extrusion_distance_y'] = 0.0


def handleSelection(objects_dict, vis, main_window):
    for object_id, obj_info in objects_dict.items():

        if obj_info.get('highlighted', False):  # Check if the 'highlighted' key exists and is True
            obj_info['selected'] = True
            obj = obj_info['object']  # Correctly reference the Open3D object
            obj.paint_uniform_color(selectedColor)  # Paint the object darker green
            vis.update_geometry(obj)
            print(f"Object {object_id} selected")
            main_window.update_dynamic_text(f"Object selected!")
            return 1

    return 0

def handleDeselection(objects_dict, vis, main_window):
    for object_id, obj_info in objects_dict.items():

        if obj_info.get('selected', False):  # Check if the 'selected' key exists and is True
            obj_info['selected'] = False  # Mark the object as deselected
            obj_info['highlighted'] = False  # Mark the object as deselected
            obj = obj_info['object']  # Correctly reference the Open3D object
            obj.paint_uniform_color([0.5, 0.5, 0.5])  # Reset the object color to grey
            vis.update_geometry(obj)
            print(f"Object {object_id} deselected")
            main_window.update_dynamic_text(f"Object {object_id} deselected")
            
    return 0

def handle_commands(clientSocket, vis, view_control, camera_parameters, geometry_dir, history, objects_dict, counters, main_window):
    try:
        #print("running")
        # Attempt to receive data, but don't block indefinitely
        clientSocket.settimeout(0.1)  # Non-blocking with timeout
        command = getTCPData(clientSocket, len(ls_dict))

        if command:
            # Parse and handle the command
            parseCommand(command, view_control, camera_parameters, vis, geometry_dir, history, objects_dict, counters, main_window)
            vis.poll_events()
            vis.update_renderer()
            # main_window.update_dynamic_text(command)
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


    mesh = o3d.geometry.TriangleMesh.create_box(width=0.2, height=0.4, depth=0.2)
    mesh.compute_vertex_normals()
    
    

    
    
    mesh.paint_uniform_color([0.5, 0.5, 0.5]) 

    
    
    

    # create visualizer and window.
    vis = o3d.visualization.Visualizer()
    

    vis.create_window(
        window_name="Open3D", width=width, height=height, left=50, top=50, visible=True
    )
    vis.add_geometry(mesh)

    vis.get_render_option().background_color = np.array([0.25, 0.25, 0.25])
    grid = create_grid(size=15, n=20, plane='xz', color=[0.5, 0.5, 0.5])

    # Add the grid to the visualizer
    vis.add_geometry(grid)


    
    #setup camera draw distance
    camera = vis.get_view_control()
    camera.set_constant_z_far(4500)

    objects_dict['object_1'] = {'object': mesh, 'center': mesh.get_center(), 'highlighted' : False, 'selected' : False,  'scale' : 100, 'type': "cube"}


    ls_dict = {}

    counters = {"ls":0, "pcd":0, "meshes":0}


    view_control = vis.get_view_control()

    # Initialize required dictionaries and parameters
    geometry_dir = {"counters": {"pcd": 0, "ls": 0, "mesh": 0}}
    history = {"operation": "", "axis": "", "lastVal": "", 'last_extrusion_distance_x': 0.0,'last_extrusion_distance_y': 0.0, 'total_extrusion': 0.0}

    main_window = MainWindow(vis)
    main_window.show()
    snap_isometric(vis, view_control)

    # Setup a QTimer to periodically check for new commands
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: handle_commands(clientSocket, vis, view_control, camera_parameters, geometry_dir, history, objects_dict, counters, main_window))
    timer.start(25)  # Check every 100 milliseconds
    

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
