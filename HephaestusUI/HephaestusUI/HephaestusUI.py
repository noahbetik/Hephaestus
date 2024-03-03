import open3d as o3d
import numpy as np
import keyboard
import socket as s

# ----------------------------------------------------------------------------------------
# SHORTCUTS
# ----------------------------------------------------------------------------------------
#
# Ctrl + K + C --> comment block
#
# ----------------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------------------------

def move_camera(view_control, direction, amount=1.5):
    cam_params = view_control.convert_to_pinhole_camera_parameters()
    extrinsic = np.array(cam_params.extrinsic)

    if direction == "forward":
        extrinsic[2, 3] -= round(amount)
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
        
    print(extrinsic[0,3])

    cam_params.extrinsic = extrinsic
    view_control.convert_from_pinhole_camera_parameters(cam_params,True)
    
def move_camera_v2(view_control, direction, amount=1.5):
    cam_params = view_control.convert_to_pinhole_camera_parameters()
    extrinsic = np.array(cam_params.extrinsic)

    if direction == "x":
        extrinsic[2, 3] += amount
    elif direction == "y":
        extrinsic[0, 3] += amount
    elif direction == "z":
        extrinsic[1, 3] += amount
        
    print(extrinsic[0,3])

    cam_params.extrinsic = extrinsic
    view_control.convert_from_pinhole_camera_parameters(cam_params,True)
    
def rotate_camera(view_control, axis, degrees=5):
    cam_params = view_control.convert_to_pinhole_camera_parameters()
    R = cam_params.extrinsic[:3, :3]
    t = cam_params.extrinsic[:3, 3].copy()  
    angle = np.radians(degrees)
    
    if axis == "y":
        rotation_matrix = np.array([[np.cos(angle), 0, np.sin(angle)],
                                    [0, 1, 0],
                                    [-np.sin(angle), 0, np.cos(angle)]])
    elif axis == "x":
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, np.cos(angle), -np.sin(angle)],
                                    [0, np.sin(angle), np.cos(angle)]])

    R = rotation_matrix @ R
    new_extrinsic = np.eye(4)  
    new_extrinsic[:3, :3] = R  
    new_extrinsic[:3, 3] = t   

    cam_params.extrinsic = new_extrinsic  # Set the new extrinsic matrix
    view_control.convert_from_pinhole_camera_parameters(cam_params, True)
    
def axis2arr(axis, delta, alpha):
    match axis:
        case "x":
            return [delta*alpha, 0, 0]
        case "y":
            return [0, delta*alpha, 0]
        case "z":
            return [0, 0, delta*alpha]
    

def startClient():
    # should have error control
    clientSocket = s.socket(s.AF_INET, s.SOCK_STREAM)
    serverAddr = ('localhost', 444) # (IP addr, port)
    clientSocket.connect(serverAddr)

    return clientSocket

def getTCPData(sock):
    # should have error control
    data = sock.recv(1024)
    readable = data.decode()
    print("Received: ", readable)
    return readable

def closeClient(sock):
    sock.close()

# ----------------------------------------------------------------------------------------
# PARSE COMMAND
# ----------------------------------------------------------------------------------------

def parseCommand(command, view_control, camera_parameters, vis, geometry_dir, history, objectHandle):
    # FORMAT
    # [command] [subcommand]

    info = command.split(" ")

    match info[0]:
        case "cam":
            handleCam(info[1:], view_control, history)
        case "select":
            return handleSelection()
        case "create":
            handleNewGeo(info[1:], view_control)
        case "update":
            handleUpdateGeo(info[1:])
            
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

    alphaM = 1 # translation scaling factor
    alphaR = 1 # rotation scaling factor
    
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
            match history["operation"]:
                case "move": # treat zoom as relative frame z-axis translation
                    delta = subcommand[1] - history["lastVal"]
                    move_camera_v2(view_control, history["axis"], delta*alphaM)
                case "rotate":
                    delta = subcommand[1] - history["lastVal"]
                    rotate_camera(view_control, history["axis"], degrees=delta*alphaR)
        case _:
            print("INVALID COMMAND")


def handleNewGeo(subcommand, view_control, camera_parameters, vis, geometry_dir):
    match subcommand[0]:
        case "box":
            print("Creating new box with dimensions ___ at ___")
            coords = list(map(float, subcommand[1].strip("()").split(",")))
            dimensions = list(map(float, subcommand[2].strip("()").split(",")))

            # Store the current view matrix
            current_view_matrix = view_control.convert_to_pinhole_camera_parameters().extrinsic

            new_box = o3d.geometry.TriangleMesh.create_box(width=dimensions[0], height=dimensions[1], depth=dimensions[2])
            new_box.compute_vertex_normals()
            new_box.translate(np.array(coords))
            vis.add_geometry(new_box)
            name = "mesh" + str(geometry_dir["counters"]["mesh"])
            geometry_dir[name] = new_box
            geometry_dir["counters"]["mesh"] += 1
        
            # Set back the stored view matrix
            camera_parameters.extrinsic = current_view_matrix
            view_control.convert_from_pinhole_camera_parameters(camera_parameters, True)
        case "line": # line handling not implemented yet
            print("Creating new line with endpoints ___ and ___")
            points = np.array([[0,0,0], [0.1, 0.1, 0.1]])
            lines = np.array([[0, 1]])
        # case "point": # is point handling useful on its own?
        #     print("Creating new point at ___")

def handleUpdateGeo(subcommand, geometryDir, history, objectHandle):

    alphaM = 1 # translation scaling factor
    alphaR = 1 # rotation scaling factor
    alphaS = 1 # scaling scaling factor

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
                case "move": # treat zoom as relative frame z-axis translation
                    arr = axis2arr(history["axis"], delta, alphaM)
                    thisGeo.translate(np.array(arr))
                case "rotate":
                    arr = axis2arr(history["axis"], delta, alphaR)
                    R = thisGeo.get_rotation_matrix_from_axis_angle(np.array(arr))
                    thisGeo.rotate(R, center=(0, 0, 0)) # investigate how center works
                case "scale":
                    thisGeo.scale(delta*alphaS, center=thisGeo.get_center())


                    
                    
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

    in_socket = startClient()
    
    camera_parameters = o3d.camera.PinholeCameraParameters()

    width = 640
    height = 480
    focal = 0.9616278814278851

    K = [[focal * width, 0, width / 2],
         [0, focal * width, height / 2],
         [0, 0, 1]]

    camera_parameters.extrinsic = np.array([[1.204026265313826, 0, 0, -0.3759973645485034],
                                               [0, -0.397051999192357, 0,
                                                   4.813624436153903, ],
                                               [0, 0, 0.5367143925232766,
                                                   7.872266818189111],
                                               [0, 0, 0, 1]])

    camera_parameters.intrinsic.set_intrinsics(
    width=width, height=height, fx=K[0][0], fy=K[1][1], cx=K[0][2], cy=K[1][2])
    
    print("Testing mesh in Open3D...")
    #armadillo_mesh = o3d.data.ArmadilloMesh()
    #mesh = o3d.io.read_triangle_mesh(armadillo_mesh.path)
    #mesh.compute_vertex_normals()

    mesh = o3d.geometry.TriangleMesh.create_box(width=0.2, height=0.2, depth=0.2)
    mesh.compute_vertex_normals()

    # create visualizer and window.
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Open3D', width=width, height=height, left=50, top=50, visible=True)
    vis.add_geometry(mesh)

    view_control = vis.get_view_control()

    i = 1 # temp
    
    geometry_dir = {"counters":{"pcd":0, "ls":0, "mesh":0}} # may need to be changed depending on how selection works
    
    pcd_counter = 0 # number of pointclouds # remove
    ls_counter = 0 # number of linesets # remove
    mesh_counter = 0 # number of meshes # remove

    
    while True:
        # queue system? threading?
        command = getTCPData(in_socket)
        
        match command:
            case 'w':
                move_camera(view_control, 'up')
            case 's':
                move_camera(view_control, 'down')
            case 'a':
                move_camera(view_control, 'left')
            case 'd':
                move_camera(view_control, 'right')
            case 'z':
                move_camera(view_control, 'forward')
            case 'x':
                move_camera(view_control, 'backward')
            case '1':
                rotate_camera(view_control, 'x', degrees=-5)
            case '2':
                rotate_camera(view_control, 'x', degrees=5)
            case '9':
                rotate_camera(view_control, 'y', degrees=-5)
            case '0':
                rotate_camera(view_control, 'y', degrees=5)
            case 'n':
                # Store the current view matrix
                current_view_matrix = view_control.convert_to_pinhole_camera_parameters().extrinsic

                new_box = o3d.geometry.TriangleMesh.create_box(width=0.2, height=0.2, depth=0.2)
                new_box.compute_vertex_normals()
                new_box.translate(np.array([i*0.2, i*0.2, i*0.2]))
                vis.add_geometry(new_box)
                name = "mesh" + str(mesh_counter)
                geometry_dir[name] = new_box
                mesh_counter += 1
                i += 1 # temp
        
                # Set back the stored view matrix
                camera_parameters.extrinsic = current_view_matrix
                view_control.convert_from_pinhole_camera_parameters(camera_parameters, True)
            case 'o':
                # add new pointcloud
                # Store the current view matrix
                current_view_matrix = view_control.convert_to_pinhole_camera_parameters().extrinsic

                # initialize pointcloud instance.
                new_pcd = o3d.geometry.PointCloud()
                # *optionally* add initial points
                points = np.random.rand(10, 3)
                new_pcd.points = o3d.utility.Vector3dVector(points)
                # include it in the visualizer before non-blocking visualization.
                vis.add_geometry(new_pcd)
                name = "pcd" + str(pcd_counter)
                geometry_dir[name] = new_pcd
                pcd_counter += 1
        
                # Set back the stored view matrix
                camera_parameters.extrinsic = current_view_matrix
                view_control.convert_from_pinhole_camera_parameters(camera_parameters, True)
            case 'p':
                # update pointcloud
                selection = geometry_dir["pcd0"] # hardcoded, make better
                selection.points.extend(np.random.rand(10, 3))
                vis.update_geometry(selection)
            # case 'k':
            #     # create new lineset
            # case 'l':
            #     # update existing lineset
                

        vis.poll_events()
        vis.update_renderer()

    vis.destroy_window()

if __name__ == "__main__":
    main()

