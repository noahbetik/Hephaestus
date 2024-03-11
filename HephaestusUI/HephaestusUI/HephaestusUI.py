import open3d as o3d
import numpy as np
import keyboard
import socket as s
import time
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


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

predefined_extrinsics = {
    'front': np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
    'left': np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]),
    'right': np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]]),
    'topdown': np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
    'bottom': np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
    'behind': np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
} 

def snap_to_closest_plane(vis, view_control):
    cam_params = view_control.convert_to_pinhole_camera_parameters()
    current_extrinsic = cam_params.extrinsic

    #extract rotational portion of extrinsic
    current_rotation = current_extrinsic[:3, :3]
    
    #find closest match
    closest_match = None
    smallest_difference = np.inf


    for name, rotation in predefined_extrinsics.items():
        difference = np.sum(np.abs(current_rotation - rotation))
        if difference < smallest_difference:
            closest_match = name
            smallest_difference = difference
            
    print(f"Closest match: {closest_match}")
    
    updated_extrinsic = current_extrinsic.copy()
    updated_extrinsic[:3, :3] = predefined_extrinsics[closest_match]

    smooth_transition(vis, view_control, updated_extrinsic)


def snap_isometric(vis, view_control):
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
    serverAddr = ('localhost', 4444) # (IP addr, port)
    serverSocket.bind(serverAddr)
    print("Server started on " + str(serverAddr[0]) + " on port " + str(serverAddr[1]))

    return serverSocket

def makeConnection(serverSocket):
    serverSocket.listen(1) # max number of queued connections
    print("Waiting for connection...")
    clientSocket, clientAddr = serverSocket.accept()
    print("\nConnection from " + str(clientAddr[0]) + " on port " + str(clientAddr[1]))
    
    return clientSocket


def getTCPData(sock):
    # should have error control
    data = sock.recv(40)
    readable = data.decode(encoding="ascii")
    print("Received: ", readable)
    processed = readable.strip("$")
    print("Strip padding: ", processed)
    return processed


def closeClient(sock):
    sock.close()


# ----------------------------------------------------------------------------------------
# PARSE COMMAND
# ----------------------------------------------------------------------------------------


def parseCommand(
    command, view_control, camera_parameters, vis, geometry_dir, history, objectHandle
):
    # FORMAT
    # [command] [subcommand]

    info = command.split(" ")
    print(info)

    match info[0]:
        case "motion":
            if objectHandle == "":
                handleCam(info[1:], view_control, history)
                return ""
            else:
                handleUpdateGeo(info[1:], geometry_dir, history, objectHandle)
                return ""
        case "select":
            return handleSelection()
        case "create":
            handleNewGeo(info[1:], view_control, camera_parameters, vis, geometry_dir)
            return ""
        case "update":
            handleUpdateGeo(info[1:], geometry_dir, history, objectHandle)
            return ""
        case "home":
            snap_isometric(vis, view_control)
        case "snap":
            snap_to_closest_plane(vis, view_control)
        


def handleCam(subcommand, view_control, history):

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
        case "position":
            match history["operation"]:
                case "pan":
                    print("camera pan update")
                    splitCoords = subcommand[2].strip("()").split(",")
                    oldCoords = history["lastVal"].strip("()").split(",")
                    deltaY = (float(splitCoords[0]) - float(oldCoords[0])) * alphaM
                    deltaZ = (float(splitCoords[1]) - float(oldCoords[1])) * alphaM
                    move_camera_v3(view_control, [deltaY, deltaZ])

                    #delta = float(subcommand[2]) - float(history["lastVal"])
                    #move_camera_v2(view_control, history["axis"], delta * alphaM)
                case "rotate":
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


def handleUpdateGeo(subcommand, geometryDir, history, objectHandle):

    alphaM = 1  # translation scaling factor
    alphaR = 1  # rotation scaling factor
    alphaS = 1  # scaling scaling factor

    match subcommand[0]:
        case "start":
            print("starting motion")
            history["operation"] = subcommand[1]
            history["axis"] = subcommand[2]
            history["lastVal"] = subcommand[3]
        case "end":
            print("ending motion")
            history["operation"] = ""
            history["axis"] = ""
            history["lastVal"] = ""
        case "position":
            print("update position")
            delta = subcommand[1] - history["lastVal"]
            thisGeo = geometryDir[objectHandle]
            match history["operation"]:
                case "move":  # treat zoom as relative frame z-axis translation
                    arr = axis2arr(history["axis"], delta, alphaM)
                    thisGeo.translate(np.array(arr))
                case "rotate":
                    arr = axis2arr(history["axis"], delta, alphaR)
                    R = thisGeo.get_rotation_matrix_from_axis_angle(np.array(arr))
                    thisGeo.rotate(R, center=(0, 0, 0))  # investigate how center works
                case "scale":
                    thisGeo.scale(delta * alphaS, center=thisGeo.get_center())
        case _:
            print("INVALID COMMAND")


def handleSelection():
    # if cursorIsOnObject
    #   return object
    # else invalid
    return


# ----------------------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------------------


def main():

    serverSocket = startServer()
    clientSocket = makeConnection(serverSocket);


    #clientSocket = startClient() #for fake TCP server
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

    # create visualizer and window.
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Open3D", width=width, height=height, left=50, top=50, visible=True
    )
    vis.add_geometry(mesh)

    view_control = vis.get_view_control()

    i = 1  # temp

    geometry_dir = {
        "counters": {"pcd": 0, "ls": 0, "mesh": 0}
    }  # may need to be changed depending on how selection works

    history = {"operation": "", "axis": "", "lastVal": ""}
    objectHandle = ""

    pcd_counter = 0  # number of pointclouds # remove
    ls_counter = 0  # number of linesets # remove
    mesh_counter = 0  # number of meshes # remove

    while True:
        # queue system? threading?
        command = getTCPData(clientSocket)

        objectHandle = parseCommand(
            command,
            view_control,
            camera_parameters,
            vis,
            geometry_dir,
            history,
            objectHandle,
        )

        vis.poll_events()
        vis.update_renderer()

    vis.destroy_window()


if __name__ == "__main__":
    main()
