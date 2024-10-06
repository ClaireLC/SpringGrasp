import numpy as np
import torch
import open3d as o3d

from utils.create_arrow import create_direct_arrow


def contactgrasp_metric(gpis,
                        pts,
                        ftip_pos,
                        aff_labels,
                        w_pos=1.0,
                        w_neg=1.0,
                        dp_thresh=None,
                        dist_to_use="gpis",
                        debug=None,):
    """
    Score grasps based on metric used in ContactGrasp paper.

    args:
    - gpis: Saved GPIS() fitted to data
    - pts: input point clouds (Tensor [batch_size, npoint, 3])
    - ftip_pos: fingertip positions (Tensor [batch_size, 4, 3])
    - aff_labels: per-point affordance labels (Tensor [batch_size, npoint])
    - w_pos: weight for positive points (aff_label == 1)
    - w_neg: weight for negative points (aff_label != 1)
    - dp_thresh: threshold for checking dot product between ft_pos normal and pt normal
    - dist_to_use: distance to use for computing cost ["gpis", "euclidean"]
    - debug: if True, show debug visualizations in o3d
    """

    if debug is None:
        debug = False

    # Check that inputs are batched (have one extra dimension)
    assert pts.ndim == 3
    assert ftip_pos.ndim == 3
    assert aff_labels.ndim == 2

    batch_size = pts.shape[0] # B
    npoint = pts.shape[1] # N

    # Get distances from each point to each finger
    pt_to_finger_dists = torch.cdist(pts, ftip_pos) # [B, npoint, 4]

    # Get distance between fingertip and nearest point on surface (SDF) 
    ft_dists, ft_vars = gpis.pred(ftip_pos) # [B, 4]

    # Get normals at the fingertip positions and surface points
    ftip_normals = gpis.compute_normal(ftip_pos) # [B, 4, 3]
    pt_normals = gpis.compute_normal(pts) # [B, npoint, 3]

    # Initialize cost to 0 for each batch (feasible grasp)
    cost = torch.zeros(batch_size) # [B,]

    # Iterate over each surface point
    for i in range(npoint):
        # Check if affordance mask or not
        label_i = aff_labels[:, i] # [B]

        # Closest fingertip id to point i
        f_i = torch.argmin(pt_to_finger_dists[:, i], dim=1)

        # Get position of the closest fingertip for each batch
        closest_ftip_pos = ftip_pos[torch.arange(batch_size), f_i] # [B, 3]

        if dist_to_use == "gpis": # SDF (e.g. ContactGrasp)
            dist = ft_dists[torch.arange(batch_size), f_i] # [B]
        elif dist_to_use == "euclidean": # Euclidean
            dist = pt_to_finger_dists[torch.arange(batch_size), i, f_i] # [B]
        else:
            raise ValueError

        # Normal vector from finger tip
        ftip_normal = ftip_normals[torch.arange(batch_size), f_i] # [B, 3]
        
        # Normal vector from surface point
        pt_normal = pt_normals[torch.arange(batch_size), i] # [B, 3]

        # Cosine similarity
        normal_dp = torch.einsum('ij, ij->i', ftip_normal, pt_normal) # [B]

        # Handle positive and negative points
        pos_mask = label_i == 1
        neg_mask = ~pos_mask

        # Initialize costs
        cost_i = torch.zeros(batch_size).double()

        # Costs for positive points
        cost_i[pos_mask] = w_pos * dist[pos_mask]**2

        # Costs for negative points
        if dp_thresh is not None:
            valid_neg_mask = torch.abs(normal_dp) >= dp_thresh

            # Only for negative points where the condition is met
            valid_neg_mask &= neg_mask
            cost_i[valid_neg_mask] = -1 * w_neg * dist[valid_neg_mask]**2
        else:
            cost_i[neg_mask] = -1 * w_neg * dist[neg_mask]**2

        cost += cost_i

        # Visualize
        if debug:
            GRASPS_TO_VIZ = list(range(batch_size))
            for g_i in GRASPS_TO_VIZ:
                print(f"Grasp {g_i} point {i}")
                print(f"ftip dist: {dist[g_i]} | normal dp: {normal_dp[g_i]}")

                pcd = o3d.geometry.PointCloud()
                colors = np.zeros_like(pts[g_i])
                color_code = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0]])

                # Draw all fingertip positions, in RGBY
                pts_to_viz = np.concatenate((pts[g_i], ftip_pos[g_i]), axis=0)
                colors = np.concatenate((
                    colors,
                    color_code,
                ), axis=0)

                pcd.points = o3d.utility.Vector3dVector(pts_to_viz)
                pcd.colors = o3d.utility.Vector3dVector(colors)

                # Draw closest ftip pos as sphere
                vis_ftip = o3d.geometry.TriangleMesh.create_sphere(radius=0.0025)
                vis_ftip.paint_uniform_color(color_code[f_i[g_i]])
                vis_ftip.translate(closest_ftip_pos[g_i])

                # Draw current point as pink sphere
                vis_pi = o3d.geometry.TriangleMesh.create_sphere(radius=0.0025)
                vis_pi.paint_uniform_color([1, 0, 1])
                vis_pi.translate(pts[g_i, i])

                # Draw pt_normal
                vis_pt_normal = create_direct_arrow(pts[g_i, i], pts[g_i, i] + pt_normal[g_i] * 0.01)
                vis_pt_normal.paint_uniform_color([1, 0, 1])

                # Draw ftip_normal
                vis_ftip_normal = create_direct_arrow(closest_ftip_pos[g_i], closest_ftip_pos[g_i] + ftip_normal[g_i] * 0.01)
                vis_ftip_normal.paint_uniform_color(color_code[f_i[g_i]])

                o3d.visualization.draw_geometries([pcd, vis_ftip, vis_pi, vis_pt_normal, vis_ftip_normal])

    return cost

