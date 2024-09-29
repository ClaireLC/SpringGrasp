"""
Get initial conditions for optimizer:
    - wrist pose
    - joint angles
    - compliance
    - ftip target pos
Use for sim.
"""
import sys, os
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import itertools
import open3d as o3d

import viz_utils as v_utils
from spring_grasp_planner.initial_guesses import WRIST_OFFSET


def visualize_point_cloud(pcd, 
                          normals=None, 
                          points=None, 
                          vt_vectors=None, 
                          center=None):
    """
    Visualize the point cloud along with optional normals, points, and vt vectors.

    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        normals (list of np.array): List of normal vectors to visualize.
        points (list of np.array): List of points to visualize.
        vt_vectors (list of np.array): List of principal directions from SVD to visualize.
        center (np.array): The center point from which vt vectors originate.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    # Compute and add the axis-aligned bounding box to the visualizer
    aabb = pcd.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)
    vis.add_geometry(aabb)
    
    if normals and points:
        for normal, point in zip(normals, points):
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector([point, point + 0.05 * normal]),
                lines=o3d.utility.Vector2iVector([[0, 1]]),)
            colors = [[1, 0, 0]]  # Red color for normal vector
            line_set.colors = o3d.utility.Vector3dVector(colors)
            vis.add_geometry(line_set)
    
    if vt_vectors and center is not None:
        colors = [[0, 1, 0], [0, 0, 1], [1, 1, 0]]  # Green, Blue, Yellow
        # Yellow is the normal vector
        for vt_vector, color in zip(vt_vectors, colors):
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector([center, center + 0.05 * vt_vector]),
                lines=o3d.utility.Vector2iVector([[0, 1]]),)
            line_set.colors = o3d.utility.Vector3dVector([color])
            vis.add_geometry(line_set)
    
    # Create a plane using the two vectors and the center
    plane_points = [
        center - 0.05 * vt_vectors[0] - 0.05 * vt_vectors[1],
        center + 0.05 * vt_vectors[0] - 0.05 * vt_vectors[1],
        center + 0.05 * vt_vectors[0] + 0.05 * vt_vectors[1],
        center - 0.05 * vt_vectors[0] + 0.05 * vt_vectors[1]]
    plane = o3d.geometry.TriangleMesh()
    plane.vertices = o3d.utility.Vector3dVector(plane_points)
    plane.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
    plane.paint_uniform_color([0.7, 0.7, 1.0])
    plane.compute_vertex_normals()
    vis.add_geometry(plane)

    vis.run()
    vis.destroy_window()


def get_default_wrist_pose(pcd):
    center = pcd.get_axis_aligned_bounding_box().get_center()
    WRIST_OFFSET[:,0] += center[0]
    WRIST_OFFSET[:,1] += center[1]
    WRIST_OFFSET[:,2] += 2 * center[2]

    return WRIST_OFFSET

def get_init_wrist_pose_from_pcd(pcd,
                                 npz_info=False, 
                                 viz=False, 
                                 num_xy_offset=1, 
                                 check_palm_ori=False,):
    """
    Get initial wrist poses.

    args:
        pcd (pcd): input point cloud
        viz (bool): if True, show debug prints and visualizations
        num_xy_offset (int): number of offsets to use along the x and y axes of the wrist frame

    returns:
        wrist_poses (np.array): [B, 6] array of B wrist poses
    """
    TEMP_VIZ = True
    wrist_poses = []

    # Handle case for using affordances
    # TODO: Use affordances
    vis_affordance = False
    if npz_info != False:
        print("Visualize Affordances")
        # TODO: Use Affordance for bounding box and plane convergence
        pts = np.asarray(pcd.points)
        affordance_pts = pts[npz_info["aff_labels"]]
        affordance_pcd = o3d.geometry.PointCloud()
        affordance_pcd.points = o3d.utility.Vector3dVector(affordance_pts)

        aabb_affordance = affordance_pcd.get_axis_aligned_bounding_box()
        center_affordance = aabb_affordance.get_center() # Center coordinates
        extent_affordance = aabb_affordance.get_extent()

        pts_affordance = np.asarray(affordance_pcd.points)
        u_aff, s_aff, vt_aff = np.linalg.svd(pts_affordance - pts_affordance.mean(axis=0), full_matrices=False)
        normal_affordance = vt_aff[-1] 

        if vis_affordance != False:
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            affordance_pcd.paint_uniform_color([1.0, 0.0, 0.0]) 
            vis.add_geometry(affordance_pcd)
            vis.run()
            vis.destroy_window()
        
        if TEMP_VIZ:
            print("Viz Affordance Normals and vt Vectors")
            vt_vectors_aff = [vt_aff[i] for i in range(3)]
            visualize_point_cloud(affordance_pcd, 
                                normals=[normal_affordance], 
                                points=[center_affordance], 
                                vt_vectors=vt_vectors_aff, 
                                center=center_affordance)

    # sys.exit()

    aabb_full_pcd = pcd.get_axis_aligned_bounding_box()
    center_full_pcd = aabb_full_pcd.get_center() # Center coordinates
    extent_full_pcd = aabb_full_pcd.get_extent() # Height, width, length

    # Fit plane and extracts normal from entire point cloud
    # Original code for full object point cloud
    pts = np.asarray(pcd.points) # Convert point cloud to np.array
    u, s, vt = np.linalg.svd(pts - center_full_pcd, full_matrices=False)
    normal_full_pcd = vt[-1] # Unit length, normal vector

    # vt[0]: The direction of the first principal component (largest variance).
    # vt[1]: The direction of the second principal component (second largest variance).
    # vt[2]: The direction of the third principal component (least variance), which is the normal to the best-fit plane.

    # Visualization for SVD, vt vectors, and normal
    if TEMP_VIZ:
        print("Viz Full Point Cloud w/ Normals and vt Vectors")
        vt_vectors = [vt[i] for i in range(3)]
        visualize_point_cloud(pcd, 
                              normals=[normal_full_pcd], 
                              points=[center_full_pcd], 
                              vt_vectors=vt_vectors, 
                              center=center_full_pcd)
    # sys.exit()

    # TODO: Re-define normal, center to affordance extractions
    if npz_info:
        use_aff = True
    else:
        use_aff = False

    # TODO: Selects normal and center
    if use_aff:
        print("\n### USING AFFORDANCE NORMAL, CENTER\n")
        normal = normal_affordance
        center = center_affordance
        extent = extent_affordance
    else:
        print("\n### USING FULL PCD NORMAL, CENTER\n")
        normal = normal_full_pcd
        center = center_full_pcd
        extent = extent_full_pcd

    # Hybrid use of affordance mask plane normals and full pcd center
    # normal = normal_affordance
    # center = center

    normal_offset_list = [0.1]
    # normal_dirs_list = [1, -1]
    normal_dirs_list = [-1] # Only from above, z-axis is flipped over
    base_x_rot_list = [0] # degrees
    base_y_rot_list = [0] # degrees
    # num_z_rot = 25 # 15 degree increments
    # num_z_rot = 13 # 30 degree increments
    # num_z_rot = 9  # 45 degree increments

    # TODO: Physically feasible degrees
    num_z_rot = 9  # 90 degree increments
    # base_z_rot_list = np.linspace(-180, 180, num_z_rot)
    base_z_rot_list = np.linspace(0, 180, num_z_rot)

    # TODO: Remove xy translation, start at center of affordance mask
    num_xy_offset = 0

    xy_offset_list = np.linspace(0, max(extent)/2., num=num_xy_offset+1)[1:]
    xy_axes = [np.array([1, 0, 0]), np.array([0, 1, 0])]

    # Generate a list of potential wrist poses for a robotic arm, 
    # given a set of parameters and constraints
    for params in itertools.product(normal_offset_list, 
                                    normal_dirs_list,
                                    base_x_rot_list, 
                                    base_y_rot_list, 
                                    base_z_rot_list,):
        
        """
        offset: 0.1, n_dir: 1, x_rot: 0, y_rot: 0, z_rot: -180.0
        offset: 0.1, n_dir: 1, x_rot: 0, y_rot: 0, z_rot: -90.0
        offset: 0.1, n_dir: 1, x_rot: 0, y_rot: 0, z_rot: 0.0
        offset: 0.1, n_dir: -1, x_rot: 0, y_rot: 0, z_rot: -180.0
        offset: 0.1, n_dir: -1, x_rot: 0, y_rot: 0, z_rot: -90.0
        """

        # Different combinations
        offset = params[0]
        n_dir = params[1]
        x_rot = params[2]
        y_rot = params[3]
        z_rot = params[4]

        # Wrist position is along plane normal
        base_pos = center + offset * normal * n_dir
        print_angles = True
        if print_angles: 
            print(f"{x_rot}, {y_rot}, {z_rot}")
            # print(f"Base pose: normal offset {n_dir*offset} | rotation ({x_rot}, {y_rot}, {z_rot})")
        # sys.exit()
        
        # Define wrist orientation so palm in perpendicular to plane normal
        # ie. in SpringGrasp convention, z axis is along plane normal
        R = np.zeros((3,3)) # Rotation matrix
        R[:, 2] = normal * n_dir # z-axis aligned with normal (1, -1)

        # print(f"Rotational matrix: {R}")

        # Rotate about x and y axis
        base_r = Rotation.from_matrix(R)
        xy_delta_r = Rotation.from_euler("xyz", [x_rot, y_rot, z_rot], degrees=True)
        # rotate base_r by xy_delta_r in its local current frame (post-multiplication)
        XYZ_ori = (base_r * xy_delta_r).as_euler("XYZ")

        # print(f"base_r: {base_r}")
        # print(f"XYZ_ori: {XYZ_ori}")

        # Check if palm is facing in orientation that will be easy for arm to reach
        # (ie. facing towards ground)
        if check_palm_ori:
            # Check cosine similaty between world frame z axis and palm z axis
            z_wf = np.array([0,0,1])
            wrist_z = base_r.as_matrix()[:, 2]
            cos_sim = np.dot(z_wf, wrist_z) / (np.linalg.norm(wrist_z))
            # print(cos_sim)
            # Want cosine similarity to be above some threshold
            if cos_sim < -0.5: continue

        # Create and append pose with base_pos (not translated along x and y) to list
        # if TEMP_VIZ: 
        #     print(f"Pose {len(wrist_poses)} xy offset from base pos: none")

        pose = np.concatenate((base_pos, XYZ_ori))
        # print(pose)
        # sys.exit()

        wrist_poses.append(pose)

        # Visualize local wrist frame
        if viz:
            v_utils.vis_wrist_pose(pcd, pose, draw_frame=True)

        # Translate along x and y axes
        if num_xy_offset > 0:
            print("### XY AXIS TRANSLATION EXECUTED")
            for trans_params in itertools.product(xy_axes, 
                                                  xy_offset_list, 
                                                  normal_dirs_list,):
                axis = trans_params[0]
                xy_offset = trans_params[1] # XY translation
                trans_dir = trans_params[2]
                H_robot_to_world = np.zeros((4, 4))
                H_robot_to_world[:3, :3] = base_r.as_matrix()
                H_robot_to_world[:3, 3] = base_pos
                H_robot_to_world[3, 3] = 1
                pos_rf = xy_offset * axis * trans_dir # Pos in local frame
                pos = (H_robot_to_world @ np.append(pos_rf, 1))[:3]

                # Create and append pose to list
                if viz: print(f"Pose {len(wrist_poses)} xy offset from base pos:", 
                              pos_rf)
                    
                pose = np.concatenate((pos, XYZ_ori))
                wrist_poses.append(pose)

                # Visualize local wrist frame
                if viz: v_utils.vis_wrist_pose(pcd, pose, draw_frame=True)

    if len(wrist_poses) == 0: raise ValueError("No initial conditions found")

    # print(wrist_poses)
    print("Number of Initial Wrist Poses: ", len(wrist_poses))
    # sys.exit()

    return np.array(wrist_poses)

def get_start_and_target_ftip_pos(wrist_poses, joint_angles, optimizer, device):
    """
    Get start and target fingertip positions for each 
    of the B wrist poses in wrist_poses

    args:
        wrist_poses (np.array): [B, 6] array of B poses. 
        joint_angles (Tensor): [B, 16] Tensor of B initial joint angles
        optimizer: optimizer with FK function

    returns:
        start_ftip_pos (Tensor): [B, 4, 3] of B start fingertip positions
        target_pose (Tensor): [B, 4, 3] of B target fingertip positions
    """
    # Based on the urdf, joint angles etc, the forward kinematics computes
    # the default finger tip position (pre pre-grasp)
    start_ftip_pos = optimizer.forward_kinematics(joint_angles, 
                                                  torch.from_numpy(wrist_poses).to(device))
    target_pose = start_ftip_pos.mean(dim=1, keepdim=True).repeat(1,4,1)
    target_pose = target_pose + (start_ftip_pos - target_pose) * 0.2
    return start_ftip_pos, target_pose