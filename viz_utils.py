import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation
import argparse
import torch
from tqdm import tqdm
import os, sys
import matplotlib

from utils.create_arrow import create_direct_arrow

def are_point_clouds_equal(pcd1, pcd2):
    points1 = np.asarray(pcd1.points)
    points2 = np.asarray(pcd2.points)
    
    # Check if the shapes of the point arrays are the same
    if points1.shape != points2.shape:
        return False
    
    # Check if all points are equal
    return np.allclose(points1, points2)


def load_affordance_pcd(pcd_path):
    loaded_data = np.load(pcd_path, allow_pickle=True)
    data = loaded_data['data'].item()

    pts_wf = data["pts_wf"]
    aff_labels = data["aff_labels"]
    pts_rgb = data["pts_rgb"]

    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_wf)

    # Create a colormap for the affordance labels
    unique_labels = np.unique(aff_labels)
    label_colors = matplotlib.colormaps.get_cmap('Set2')

    # Map each label to a color
    label_color_map = np.array([label_colors(i) for i in range(len(unique_labels))])[:, :3]
    label_colors_rgb = label_color_map[aff_labels.astype(int)]

    # Set the colors of the point cloud based on affordance labels
    pcd.colors = o3d.utility.Vector3dVector(label_colors_rgb)
    
    return pcd


def sanity_check_intersection_points(intersection_points, pcd, threshold=0.003):
    original_points = np.asarray(pcd.points)
    valid_points = []
    for point in intersection_points:
        distances = np.linalg.norm(original_points - point, axis=1)
        min_distance = np.min(distances)
        if min_distance < threshold:
            valid_points.append(point)
        else:
            print(f"Point {point} is not close to any point in the original point cloud.")

    return valid_points

def check_vector_intersection_points(arrow_start, arrow_end, pcd):
    # Estimate normals for the point cloud
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # Convert point cloud to mesh using BPA
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3 * avg_dist
    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector([radius, radius * 2])
    )
    
    # Compute direction and length of the arrow
    direction = arrow_end - arrow_start
    length = np.linalg.norm(direction)
    direction /= length
    
    # Create rays from the arrow start to the arrow end
    rays = o3d.core.Tensor([[*arrow_start, *direction]], dtype=o3d.core.Dtype.Float32)
    
    # Convert mesh to Open3D tensor
    bpa_mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(bpa_mesh)
    
    # Perform ray casting
    raycasting_scene = o3d.t.geometry.RaycastingScene()
    mesh_id = raycasting_scene.add_triangles(bpa_mesh_t)
    ans = raycasting_scene.cast_rays(rays)
    
    # Check if there is an intersection
    hit = ans['t_hit'].isfinite()
    if not hit.any().item():
        return False, None  # Return False if no intersection is found
    
    # Compute intersection points manually
    t_hit = ans['t_hit'][hit].numpy()
    intersection_points_mesh = arrow_start + t_hit[:, None] * direction
    
    # Find the corresponding points in the original point cloud
    original_points = np.asarray(pcd.points)
    intersection_points_pcd = []
    for point in intersection_points_mesh:
        distances = np.linalg.norm(original_points - point, axis=1)
        closest_point_index = np.argmin(distances)
        intersection_points_pcd.append(original_points[closest_point_index])
    
    return True, intersection_points_pcd  # Return True and the list of intersection points


def check_arrow_contact_with_pcd(arrow_start, arrow_end, pcd):
    # Estimate normals for the point cloud
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # Convert point cloud to mesh using BPA
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3 * avg_dist
    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector([radius, radius * 2])
    )
    
    # Compute direction and length of the arrow
    direction = arrow_end - arrow_start
    length = np.linalg.norm(direction)
    direction /= length
    
    # Create rays from the arrow start to the arrow end
    rays = o3d.core.Tensor([[*arrow_start, *direction]], dtype=o3d.core.Dtype.Float32)
    
    # Convert mesh to Open3D tensor
    bpa_mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(bpa_mesh)
    
    # Perform ray casting
    raycasting_scene = o3d.t.geometry.RaycastingScene()
    raycasting_scene.add_triangles(bpa_mesh_t)
    ans = raycasting_scene.cast_rays(rays)
    
    # Check if there is an intersection
    hit = ans['t_hit'].isfinite()
    return hit.any().item()  # Return True if any intersection is found


def vis_grasp(tip_pose, target_pose):
    # print("=============================================")
    # print(type(tip_pose))
    # print(type(target_pose))
    # # sys.exit()
    # print("=============================================")
    # print(tip_pose.shape) # torch.Size([4, 3])
    # print(target_pose.shape) # torch.Size([4, 3])
    # sys.exit()

    if torch.is_tensor(tip_pose):
        tip_pose = tip_pose.cpu().detach().numpy().squeeze()
    if torch.is_tensor(target_pose):
        target_pose = target_pose.cpu().detach().numpy().squeeze()
    # print("=============================================")
    # print(tip_pose.shape) # (4, 3)
    # print(target_pose.shape) # (4, 3)

    tips = []
    targets = []
    arrows = []
    color_code = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0]]) # for arrows

    # Loop through each finger pair (4 in total)
    for i in range(4):
        # Draw tip
        tip = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)
        tip.paint_uniform_color(color_code[i])
        tip.translate(tip_pose[i])
        # Draw target
        target = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)
        target.paint_uniform_color(color_code[i] * 0.4)
        target.translate(target_pose[i])
        # Create arrow point from tip to target
        arrow = create_direct_arrow(tip_pose[i], target_pose[i])
        arrow.paint_uniform_color(color_code[i])
        tips.append(tip)
        targets.append(target)
        arrows.append(arrow)

    return tips, targets, arrows

def vis_wrist_pose(pcd, pose, draw_frame=False, wrist_frame="springgrasp",):
    geoms_list = [pcd]

    # Get wrist ref frame
    mesh_wrist = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
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
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, 
                                                                       origin=[0,0,0])
        geoms_list.append(mesh_frame)

    o3d.visualization.draw_geometries(geoms_list)

def vis_results(pcd,
                init_ftip_pos,
                target_ftip_pos,
                draw_frame=False,
                wrist_pose=None, # not none
                wrist_frame="springgrasp",
                save_path=None,):
    
    # Load affordance mask
    pcd_path = "/juno/u/junhokim/code/zed_redis/pcd_data/plier/obj8/ann_gt_pcd.npz"
    pcd_n = load_affordance_pcd(pcd_path)

    # Check if the passed pcd is the same as the loaded pcd but with different colors
    if are_point_clouds_equal(pcd, pcd_n):
        print("The passed point cloud and the loaded point cloud are the same, but with different colors.")
    else:
        print("The passed point cloud and the loaded point cloud are different.")
    
    pcd = pcd_n
    
    # sys.exit()

    # Plot and save without opening a window
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Get geometries to visualize grasp
    tips, targets, arrows = vis_grasp(init_ftip_pos, target_ftip_pos)
    # print("=============================================")
    # print(type(tips), type(targets), type(arrows)) # <class 'list'>
    # print("=============================================")
    # print(tips) # TriangleMesh with 762 points and 1520 triangles
    # print("=============================================")
    # print(targets) # TriangleMesh with 762 points and 1520 triangles
    # print("=============================================")
    # print(arrows) # TriangleMesh with 124 points and 240 triangles

    # Check if any arrow makes contact with the point cloud
    # for i in range(len(arrows)):
    #     arrow_start = init_ftip_pos[i].cpu().detach().numpy() if torch.is_tensor(init_ftip_pos[i]) else init_ftip_pos[i]
    #     arrow_end = target_ftip_pos[i].cpu().detach().numpy() if torch.is_tensor(target_ftip_pos[i]) else target_ftip_pos[i]
    #     if check_arrow_contact_with_pcd(arrow_start, arrow_end, pcd):
    #         print(f"Arrow {i} makes contact with the point cloud.")
    #     else:
    #         print(f"Arrow {i} does not make contact with the point cloud.")

    intersection_points_all = []
    for i in range(len(arrows)):
        arrow_start = init_ftip_pos[i].cpu().detach().numpy() if torch.is_tensor(init_ftip_pos[i]) else init_ftip_pos[i]
        arrow_end = target_ftip_pos[i].cpu().detach().numpy() if torch.is_tensor(target_ftip_pos[i]) else target_ftip_pos[i]
        contact, intersection_points = check_vector_intersection_points(arrow_start, arrow_end, pcd)
    
        if contact:
            print(f"Arrow {i} makes contact with the point cloud at points: {intersection_points}.")
            intersection_points_all.extend(intersection_points)
        else:
            print(f"Arrow {i} does not make contact with the point cloud.")
    
    # Sanity check for intersection points
    valid_intersection_points = sanity_check_intersection_points(intersection_points_all, pcd)
    print(len(valid_intersection_points))


    #! TRY LOADING THE MASK HERE AND VALIDATE WITH THE INTERSECTION POINTS


    # Create spheres for intersection points
    intersection_spheres = []
    for point in intersection_points_all:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)
        sphere.paint_uniform_color([1, 0, 0])  # Color the intersection points red
        sphere.translate(point)
        intersection_spheres.append(sphere)

    # Geometries are added to the visualizer in a loop
    geoms_list = [pcd, *tips, *targets, *arrows, *intersection_spheres]
    for g in geoms_list:
        vis.add_geometry(g)

    # Draw reference frame
    if draw_frame:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, 
                                                                       origin=[0,0,0])
        vis.add_geometry(mesh_frame)

    # Draw wrist
    if wrist_pose is not None:
        mesh_wrist = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        wrist_pos = wrist_pose[:3]
        wrist_ori_XYZ = wrist_pose[3:]

        _wrist_R = Rotation.from_euler("XYZ", wrist_pose[3:])
        if wrist_frame == "springgrasp":
            wrist_R = (_wrist_R).as_matrix()
        else:
            # Transform from this wrist rotation to original local hand frame
            transform = Rotation.from_euler("xyz", [0, 90, 0], degrees=True)
            wrist_R = (_wrist_R*transform).as_matrix()

        mesh_wrist.translate(wrist_pos)
        mesh_wrist.rotate(wrist_R)
        vis.add_geometry(mesh_wrist)
    
        ## Set-up & customize initial view
        # Move viewing camera
        ctr = vis.get_view_control()
        fov = ctr.get_field_of_view()
        param = ctr.convert_to_pinhole_camera_parameters()
        H = np.eye(4)
        # if cam_frame_x_front:
        #    # Camera frame that points are in has +x facing viewing direction
        #    # Rotate open3d viz camera accordingly
        cam_R = Rotation.from_euler("XYZ", [0, 180, 0], degrees=True).as_matrix()
        H[:3, :3] = (_wrist_R.as_matrix() @ cam_R).T
        H[:3, 3] = -H[:3, :3] @ wrist_pos
        H[2, 3] += 0.3  # Move camera back
        param.extrinsic = H
        ctr.convert_from_pinhole_camera_parameters(param)
        _param = ctr.convert_to_pinhole_camera_parameters()
    
    vis.poll_events() # process window events
    vis.update_renderer() # updates window

    if save_path is None:
        vis.run()
    else:
        vis.capture_screen_image(save_path, do_render=True,)
    vis.destroy_window()


def main(args):
    grasp_dict = np.load(args.grasp_path, allow_pickle=True)["data"].item() 

    # Create save dir
    if args.save:
        grasp_name = os.path.splitext(os.path.basename(args.grasp_path))[0]
        save_dir = os.path.join(os.path.dirname(args.grasp_path), f"{grasp_name}_img")
        if not os.path.exists(save_dir): os.makedirs(save_dir)

    # Visualize only the feasible indices
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
            draw_frame=(save_path is None),
            wrist_pose=palm_pose,
            wrist_frame="original",
            save_path=save_path,
        )

if __name__ == "__main__":
    """
    python viz_utils.py ./data/plier/opt_fmn-none_fcd-euclidean_ffp-pregrasp/sg_predictions.npz
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "grasp_path",
        type=str,
        help="Path to .npz file with grasp optimization results",
    )
    parser.add_argument("--save", "-s", action="store_true")
    args = parser.parse_args()
    main(args)