import numpy as np
import torch
import open3d as o3d

from utils.create_arrow import create_direct_arrow

def contactgrasp_metric(
    gpis,
    pts,
    ftip_pos,
    aff_labels,
    w_pos=1.0,
    w_neg=1.0,
    dp_thresh=None,
    dist_to_use="gpis",
    debug=False,
):
    # Score grasps based on metric used in ContactGrasp paper

    cost = 0

    #tree = KDTree(ftip_pos)
    ## neighbor_indices: finger idx closest to each point in pts
    ## neighbor_dists: distance between closest ftip pos and each point in pts
    #neighbor_dists, neighbor_indices = tree.query(pts)

    # Get distances from each point to each finger
    pt_to_finger_dists = torch.cdist(torch.tensor(pts), torch.tensor(ftip_pos)) # [npoint, 4]

    ft_dists, ft_vars = gpis.pred(torch.tensor(ftip_pos))

    # Get SDF gradients at all points and fingertip positions
    ftip_normals = gpis.compute_normal(torch.tensor(ftip_pos)).detach().cpu().numpy()
    pt_normals = gpis.compute_normal(torch.tensor(pts)).detach().cpu().numpy()


    for i in range(pts.shape[0]):
        label_i = aff_labels[i]

        # Closest fingertip id to point i
        #f_i = neighbor_indices[i]
        f_i = torch.argmin(pt_to_finger_dists[i])
        closest_ftip_pos = ftip_pos[f_i]

        if dist_to_use == "gpis":
            # This is what ContactGrasp did
            dist =  ft_dists[f_i] # SDF value of closest fingertip to point (used in ContactGrasp)
        elif dist_to_use == "euclidean":
            dist =  pt_to_finger_dists[i][f_i] # Distance of closest fingertip to current point
        else:
            raise ValueError

        ftip_normal = ftip_normals[f_i]
        pt_normal = pt_normals[i]
        
        normal_dp = np.dot(ftip_normal, pt_normal)

        if label_i == 1:
            # Positive point
            cost_i = w_pos * dist.item()**2
        else:
            # Negative point
            if dp_thresh is not None:
                if abs(normal_dp) >= dp_thresh:
                    cost_i = -1 * w_neg * dist.item()**2
                else: 
                    cost_i = 0
            else:
                cost_i = -1 * w_neg * dist.item()**2
        
        cost += cost_i

        if debug and i % 100 == 0:
            print("Point", i)
            print(f"ftip dist: {dist} | normal dp: {normal_dp}")

            pcd = o3d.geometry.PointCloud()
            colors = np.zeros_like(pts)
            color_code = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0]])

            # Draw all fingertip positions, in RGBY
            pts_to_viz = np.concatenate((pts, ftip_pos), axis=0)
            colors = np.concatenate((
                colors,
                color_code,
            ), axis=0)

            pcd.points = o3d.utility.Vector3dVector(pts_to_viz)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            # Draw closest ftip pos as sphere
            vis_ftip = o3d.geometry.TriangleMesh.create_sphere(radius=0.0025)
            vis_ftip.paint_uniform_color(color_code[f_i])
            vis_ftip.translate(closest_ftip_pos)

            # Draw current point as pink sphere
            vis_pi = o3d.geometry.TriangleMesh.create_sphere(radius=0.0025)
            vis_pi.paint_uniform_color([1, 0, 1])
            vis_pi.translate(pts[i])

            # Draw pt_normal
            vis_pt_normal = create_direct_arrow(pts[i], pts[i] + pt_normal * 0.01)
            vis_pt_normal.paint_uniform_color([1, 0, 1])

            # Draw ftip_normal
            vis_ftip_normal = create_direct_arrow(closest_ftip_pos, closest_ftip_pos + ftip_normal * 0.01)
            vis_ftip_normal.paint_uniform_color(color_code[f_i])

            o3d.visualization.draw_geometries([pcd, vis_ftip, vis_pi, vis_pt_normal, vis_ftip_normal])

    return cost

