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

def parseCommand(command, view_control, camera_parameters, vis, geometry_dir):
    # FORMAT:
    # [TYPE] [SUBTYPE] [(COORDS1)]* [(COORDS2)]* [ID]* [(DIMENSIONS)]*
    # * -- optional
    # eg1 "cam rotate (20,0,-50)" -- rotate camera 20deg about x-axis and -50deg about z-axis
    # eg2 "create box (1,2,3) (0.4,0.8,0.4) -- create new box mesh at (1, 2, 3) with dimensions 0.4*0.8*0.4

    info = command.split(" ")

    match info[0]:
        case "cam":
            handleCam(info[1:], view_control)
        case "create":
            handleNewGeo(info[1:], view_control)
        case "update":
            handleUpdateGeo(info[1:])
            
def handleCam(subcommand, view_control):
    vals = list(map(float, subcommand[1].strip("()").split(","))) # isolate dimensions and convert to float
    
    match subcommand[0]:
        case "move":
            # taking current reference frame: 
            # x = zoom in/out
            # y = left/right 
            # z = up/down

            print("moving by " + str(subcommand[1]))
            if (vals[0] != 0):
                move_camera_v2(view_control, 'x', vals[0])
            if (vals[1] != 0):
                move_camera_v2(view_control, 'y', vals[1])
            if (vals[2] != 0):
                move_camera_v2(view_control, 'z', vals[2])
                
        case "rotate":
            print("rotating by " + str(subcommand[1]))
            if (vals[0] != 0):
                rotate_camera(view_control, 'x', degrees=vals[0])
            if (vals[1] != 0):
                rotate_camera(view_control, 'y', degrees=vals[1])
            # if (dimensions[2] != 0): # z rotation not implemented yet
            #     rotate_camera(view_control, 'z', degrees=dimensions[2])

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

def handleUpdateGeo(subcommand):
    return ""

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
    
    geometry_dir = {"counters":{"pcd":0, "ls":0, "mesh":0}} # put all new geo objects here so we have handle to update
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

