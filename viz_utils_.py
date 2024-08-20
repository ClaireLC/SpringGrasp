import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation
import argparse
import torch
from tqdm import tqdm
import os

from utils.create_arrow import create_direct_arrow

def vis_grasp(tip_pose, target_pose):
    if torch.is_tensor(tip_pose):
        tip_pose = tip_pose.cpu().detach().numpy().squeeze()
    if torch.is_tensor(target_pose):
        target_pose = target_pose.cpu().detach().numpy().squeeze()
    tips = []
    targets = []
    arrows = []
    color_code = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0]])
    for i in range(4):
        tip = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        tip.paint_uniform_color(color_code[i])
        tip.translate(tip_pose[i])
        target = o3d.geometry.TriangleMesh.create_sphere(radius=0.0025)
        target.paint_uniform_color(color_code[i] * 0.4)
        target.translate(target_pose[i])
        # create arrow point from tip to target
        arrow = create_direct_arrow(tip_pose[i], target_pose[i])
        arrow.paint_uniform_color(color_code[i])
        tips.append(tip)
        targets.append(target)
        arrows.append(arrow)
    return tips, targets, arrows

def vis_wrist_pose(
    pcd,
    pose,
    draw_frame=False,
    wrist_frame="springgrasp",
):
    geoms_list = [pcd]

    # Get wrist ref frame
    mesh_wrist = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.05
    )
    _wrist_R = Rotation.from_euler("XYZ",pose[3:])
    if wrist_frame == "springgrasp":
        wrist_R = (_wrist_R).as_matrix()
    else:
        # Transform from this wrist rotation to original local hand frame
        transform = Rotation.from_euler("xyz", [0, 90, 0], degrees=True)
        wrist_R = (_wrist_R*transform).as_matrix()
    mesh_wrist.translate(pose[:3])
    mesh_wrist.rotate(wrist_R)
    geoms_list.append(mesh_wrist)

    if draw_frame:
        # Global ref frame
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )
        geoms_list.append(mesh_frame)

    o3d.visualization.draw_geometries(geoms_list)

def vis_results(
    pcd,
    init_ftip_pos,
    target_ftip_pos,
    draw_frame=False,
    wrist_pose=None,
    wrist_frame="springgrasp",
    save_path=None,
):
    # Plot and save without opening a window

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Get geometries to visualize grasp
    tips, targets, arrows = vis_grasp(init_ftip_pos, target_ftip_pos)

    geoms_list = [pcd, *tips, *targets, *arrows,]
    for g in geoms_list:
        vis.add_geometry(g)

    # Draw reference frame
    if draw_frame:
        # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        vis.add_geometry(mesh_frame)

    # Draw wrist
    if wrist_pose is not None:
        mesh_wrist = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        wrist_pos = wrist_pose[:3]
        wrist_ori_XYZ = wrist_pose[3:]

        _wrist_R = Rotation.from_euler("XYZ",wrist_pose[3:])
        if wrist_frame == "springgrasp":
            wrist_R = (_wrist_R).as_matrix()
        else:
            # Transform from this wrist rotation to original local hand frame
            transform = Rotation.from_euler("xyz", [0, 90, 0], degrees=True)
            wrist_R = (_wrist_R*transform).as_matrix()

        mesh_wrist.translate(wrist_pos)
        mesh_wrist.rotate(wrist_R)
        vis.add_geometry(mesh_wrist)
    
        # Move viewing camera
        ctr = vis.get_view_control()
        fov = ctr.get_field_of_view()
        param = ctr.convert_to_pinhole_camera_parameters()
        H = np.eye(4)
        #if cam_frame_x_front:
        #    # Camera frame that points are in has +x facing viewing direction
        #    # Rotate open3d viz camera accordingly
        cam_R = Rotation.from_euler("XYZ", [0, 180, 0], degrees=True).as_matrix()
        H[:3, :3] = (_wrist_R.as_matrix() @ cam_R).T
        H[:3, 3] = -H[:3, :3] @ wrist_pos
        H[2, 3] += 0.3  # Move camera back
        param.extrinsic = H
        ctr.convert_from_pinhole_camera_parameters(param)
        _param = ctr.convert_to_pinhole_camera_parameters()
    
    vis.poll_events()
    vis.update_renderer()

    if save_path is None:
        vis.run()
    else:
        vis.capture_screen_image(
            save_path,
            do_render=True,
        )
    vis.destroy_window()


def main(args):
    grasp_dict = np.load(args.grasp_path, allow_pickle=True)["data"].item() 

    # Create save dir
    if args.save:
        grasp_name = os.path.splitext(os.path.basename(args.grasp_path))[0]
        save_dir = os.path.join(os.path.dirname(args.grasp_path), f"{grasp_name}_img")
        if not os.path.exists(save_dir): os.makedirs(save_dir)

    if "feasible_idx" in grasp_dict:
        feasible_idx = grasp_dict["feasible_idx"]
    else:
        feasible_idx = None

    # Iterate through grasps in grasp_path.npz
    for grasp_i in tqdm(range(grasp_dict["palm_pose"].shape[0])):

        # Only visualize feasible grasps
        if not args.save:
            if feasible_idx is not None and grasp_i not in feasible_idx:
                continue
            else:
                print("Visualizing:", grasp_i)

        pts = grasp_dict["input_pts"]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        init_ftip_pos = grasp_dict["start_tip_pose"][grasp_i]
        target_ftip_pos = grasp_dict["target_tip_pose"][grasp_i]
        palm_pose = grasp_dict["palm_pose"][grasp_i]

        if args.save:
            save_name = f"grasp_{grasp_i}"
            if feasible_idx is not None and grasp_i in feasible_idx:
                save_name += "_feasible"
            save_name += ".png"
            save_path = os.path.join(save_dir, save_name)
        else:
            save_path = None

        vis_results(
            pcd,
            init_ftip_pos,
            target_ftip_pos,
            draw_frame=(save_path is None), # True
            wrist_pose=palm_pose,
            wrist_frame="original",
            save_path=save_path,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "grasp_path",
        type=str,
        help="Path to .npz file with grasp optimization results",
    )
    parser.add_argument("--save", "-s", action="store_true")
    args = parser.parse_args()
    main(args)