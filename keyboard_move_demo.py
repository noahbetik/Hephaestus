import open3d as o3d
import numpy as np
import keyboard

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

def main():
    
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
    armadillo_mesh = o3d.data.ArmadilloMesh()
    mesh = o3d.io.read_triangle_mesh(armadillo_mesh.path)
    mesh.compute_vertex_normals()

    # create visualizer and window.
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Open3D', width=width, height=height, left=50, top=50, visible=True)
    vis.add_geometry(mesh)

    view_control = vis.get_view_control()
    cam_params = view_control.convert_to_pinhole_camera_parameters()


    while not keyboard.is_pressed('esc'):
        if keyboard.is_pressed('w'):
            move_camera(view_control, 'forward')
        if keyboard.is_pressed('s'):
            move_camera(view_control, 'backward')
        if keyboard.is_pressed('a'):
            move_camera(view_control, 'left')
        if keyboard.is_pressed('d'):
            move_camera(view_control, 'right')
        if keyboard.is_pressed('up'):
            move_camera(view_control, 'up')
        if keyboard.is_pressed('down'):
            move_camera(view_control, 'down')

        vis.poll_events()
        vis.update_renderer()
        print(keyboard.is_pressed('d'))

    vis.destroy_window()

if __name__ == "__main__":
    main()
