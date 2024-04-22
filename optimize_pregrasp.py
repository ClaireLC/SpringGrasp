import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from gpis.gpis import GPIS
import torch
from utils import robot_configs
from utils.pb_grasp_visualizer import GraspVisualizer
from utils.create_arrow import create_direct_arrow
import os

from spring_grasp_planner.optimizers import FCGPISGraspOptimizer, SpringGraspOptimizer
from spring_grasp_planner.initial_guesses import WRIST_OFFSET

import viz_utils as v_utils
import get_initial_conditions as init_cond

device = torch.device("cpu")

def vis_grasp(tip_pose, target_pose):
    tip_pose = tip_pose.cpu().detach().numpy().squeeze()
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

optimizers = {"sp": SpringGraspOptimizer,
              "fc":   FCGPISGraspOptimizer}

if __name__ == "__main__":
    from argparse import ArgumentParser
    import json

    parser = ArgumentParser()
    parser.add_argument("--num_iters", type=int, default=200)
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--pcd_file", type=str, default=None)
    parser.add_argument("--mode", type=str, default="sp") # fc
    parser.add_argument("--hand", type=str, default="allegro_right", choices=["allegro", "allegro_right", "leap"])
    parser.add_argument("--mass", type=float, default=0.5) # Not use in the paper, may run into numerical issues.
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--vis_gpis", action="store_true", default=False)
    parser.add_argument("--weight_config", type=str, default=None)
    parser.add_argument("--npz_path", type=str, help="Path to input .npz file with points")
    parser.add_argument("--vis", choices=["pb", "o3d"], default="o3d", help="Visualize mode. If not specified, do not show vis.")
    parser.add_argument("--vis_ic", action="store_true", help="Visualize initial conditions")
    args = parser.parse_args()

    if args.weight_config is not None:
        weight_config = json.load(open(f"weight_config/{args.weight_config}.json"))
    else:
        weight_config = None

    if args.pcd_file is not None:
        pcd = o3d.io.read_point_cloud(args.pcd_file)
        center = pcd.get_axis_aligned_bounding_box().get_center()
        WRIST_OFFSET[:,0] += center[0]
        WRIST_OFFSET[:,1] += center[1]
        WRIST_OFFSET[:,2] += 2 * center[2]
        init_wrist_poses = WRIST_OFFSET
    elif args.npz_path is not None:
        input_dict = np.load(args.npz_path, allow_pickle=True)["data"].item() 
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(input_dict["pts_wf"])
        #init_wrist_poses = init_cond.get_default_wrist_pose(pcd)
        init_wrist_poses = init_cond.get_init_wrist_pose_from_pcd(pcd)
        center = pcd.get_axis_aligned_bounding_box().get_center()
    else:
        pcd = o3d.io.read_point_cloud("data/obj_cropped.ply")
        center = pcd.get_axis_aligned_bounding_box().get_center()
        WRIST_OFFSET[:,0] += center[0]
        WRIST_OFFSET[:,1] += center[1]
        WRIST_OFFSET[:,2] += 2 * center[2]
        init_wrist_poses = WRIST_OFFSET
    
    # GPIS formulation
    bound = max(max(pcd.get_axis_aligned_bounding_box().get_extent()) / 2 + 0.01, 0.1) # minimum bound is 0.1
    gpis = GPIS(0.08, 1)
    pcd_simple = pcd.farthest_point_down_sample(200)
    points = np.asarray(pcd_simple.points)
    points = torch.tensor(points).to(device).double()
    data_noise = [0.005] * len(points)
    weights = torch.rand(50,len(points)).to(device).double()
    weights = torch.softmax(weights * 100, dim=1)
    internal_points = weights @ points
    externel_points = torch.tensor([[-bound, -bound, -bound], 
                                    [bound, -bound, -bound], 
                                    [-bound, bound, -bound],
                                    [bound, bound, -bound],
                                    [-bound, -bound, bound], 
                                    [bound, -bound, bound], 
                                    [-bound, bound, bound],
                                    [bound, bound, bound],
                                    [-bound,0., 0.], 
                                    [0., -bound, 0.], 
                                    [bound, 0., 0.], 
                                    [0., bound, 0],
                                    [0., 0., bound], 
                                    [0., 0., -bound]]).double().to(device)
    externel_points += torch.from_numpy(center).to(device).double()
    y = torch.vstack([bound * torch.ones_like(externel_points[:,0]).to(device).view(-1,1),
                    torch.zeros_like(points[:,0]).to(device).view(-1,1),
                    -bound * 0.3 * torch.ones_like(internal_points[:,0]).to(device).view(-1,1)])
    gpis.fit(torch.vstack([externel_points, points, internal_points]), y,
                noise = torch.tensor([0.2] * len(externel_points)+
                                    data_noise +
                                    [0.05] * len(internal_points)).double().to(device))
    if args.vis_gpis:
        test_mean, test_var, test_normal, lb, ub = gpis.get_visualization_data([-bound+center[0],-bound+center[1],-bound+center[2]],
                                                                            [bound+center[0],bound+center[1],bound+center[2]],steps=100)
        plt.imshow(test_mean[:,:,50], cmap="seismic", vmax=bound, vmin=-bound)
        plt.show()
        vis_points, vis_normals, vis_var = gpis.topcd(test_mean, test_normal, [-bound+center[0],-bound+center[1],-bound+center[2]],[bound+center[0],bound+center[1],bound+center[2]],test_var=test_var,steps=100)
        vis_var = vis_var / vis_var.max()
        fitted_pcd = o3d.geometry.PointCloud()
        fitted_pcd.points = o3d.utility.Vector3dVector(vis_points)
        fitted_pcd.normals = o3d.utility.Vector3dVector(vis_normals)
        # Create color code from variance
        colors = np.zeros_like(vis_points)
        colors[:,0] = vis_var
        colors[:,2] = 1 - vis_var
        fitted_pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([fitted_pcd])

        if args.exp_name is not None:
            np.savez(f"gpis_states/{args.exp_name}_gpis.npz", mean=test_mean, var=test_var, normal=test_normal, ub=ub, lb=lb)
        else:
            gpis_dict = {
                "mean": test_mean,
                "var": test_var,
                "normal": test_normal,
                "ub": ub,
                "lb": lb,
                "bound": bound,
                "center": center,
            }
            save_path = os.path.join(os.path.dirname(args.npz_path), "sg_gpis.npz")
            print("Saving GPIS results to:", save_path)
            np.savez_compressed(save_path, data=gpis_dict)
            save_path = os.path.join(os.path.dirname(args.npz_path), "sg_gpis.ply")
            print("Saving GPIS viz pcd to:", save_path)
            o3d.io.write_point_cloud(save_path, fitted_pcd)  

    init_tip_pose = torch.tensor([[[0.05,0.05, 0.02],[0.06,-0.0, -0.01],[0.03,-0.04,0.0],[-0.07,-0.01, 0.02]]]).double().to(device)
    init_joint_angles = torch.tensor(robot_configs[args.hand]["ref_q"].tolist()).unsqueeze(0).double().to(device)
    if args.mode == "fc":
        compliance = torch.tensor([[80.0,80.0,80.0,160.0]]).to(device) * 2.0
    else:
        compliance = torch.tensor([[80.0,80.0,80.0,160.0]]).to(device)
    friction_mu = args.friction
    
    if args.hand == "leap": # Not supported in the paper.
        robot_urdf = "assets/leap_hand/robot.urdf"
    elif args.hand == "allegro": # Used in the paper
        robot_urdf = "assets/allegro_hand/allegro_hand_description_left.urdf"
    elif args.hand == "allegro_right": # Used in the paper
        robot_urdf = "assets/allegro_hand/allegro_hand_description_right.urdf"

    if args.mode == "fc":
        grasp_optimizer = optimizers[args.mode](robot_urdf,
                                                ee_link_names=robot_configs[args.hand]["ee_link_name"],
                                                ee_link_offsets=robot_configs[args.hand]["ee_link_offset"].tolist(),
                                                anchor_link_names=robot_configs[args.hand]["collision_links"],
                                                anchor_link_offsets=robot_configs[args.hand]["collision_offsets"].tolist(),
                                                collision_pairs=robot_configs[args.hand]["collision_pairs"],
                                                ref_q = robot_configs[args.hand]["ref_q"].tolist(),
                                                optimize_target=True,
                                                optimize_palm=True, # NOTE: Experimental
                                                num_iters=args.num_iters,
                                                palm_offset=init_wrist_poses,
                                                uncertainty=20.0,
                                                # Useless for now
                                                mass=args.mass, 
                                                com=[args.com_x,args.com_y,args.com_z],
                                                gravity=False)
    elif args.mode == "sp":
        grasp_optimizer = optimizers[args.mode](robot_urdf,
                                                ee_link_names=robot_configs[args.hand]["ee_link_name"],
                                                ee_link_offsets=robot_configs[args.hand]["ee_link_offset"].tolist(),
                                                anchor_link_names=robot_configs[args.hand]["collision_links"],
                                                anchor_link_offsets=robot_configs[args.hand]["collision_offsets"].tolist(),
                                                collision_pairs=robot_configs[args.hand]["collision_pairs"],
                                                ref_q = robot_configs[args.hand]["ref_q"].tolist(),
                                                optimize_target=True,
                                                optimize_palm=True, # NOTE: Experimental
                                                num_iters=args.num_iters,
                                                palm_offset=init_wrist_poses,
                                                mass=args.mass, com=center[:3],
                                                gravity=False,
                                                weight_config=weight_config)

    # Get intial conditions
    num_guesses = len(init_wrist_poses)
    print("Number of initial guesses:", num_guesses)
    init_joint_angles = init_joint_angles.repeat_interleave(num_guesses,dim=0)
    compliance = compliance.repeat_interleave(num_guesses,dim=0)
    init_start_ftip_pos, target_pose = init_cond.get_start_and_target_ftip_pos(
        init_wrist_poses, init_joint_angles, grasp_optimizer, device,
    )
    # Save initial pose info
    if args.npz_path is not None:
        data_dict = {
            "start_tip_pose": init_start_ftip_pos.cpu().detach().numpy(),
            "target_tip_pose": target_pose.cpu().detach().numpy(),
            "palm_pose": init_wrist_poses,
            "input_pts": np.asarray(pcd.points),
            "input_path": args.npz_path,
            "compliance": compliance.cpu().detach().numpy(),
            "joint_angles": init_joint_angles.cpu().detach().numpy(),
        }
        save_path = os.path.join(os.path.dirname(args.npz_path), "sg_init_cond.npz")
        print("Saving initial conditions to:", save_path)
        np.savez_compressed(save_path, data=data_dict)
    if args.vis_ic:
        for i in range(num_guesses):
            print("Initial condition:", i)
            v_utils.vis_results(
                pcd,
                init_start_ftip_pos[i],
                target_pose[i],
                wrist_pose=init_wrist_poses[i],
                draw_frame=True,
                #wrist_frame="original",
            )
        quit()

    # Run optimization
    if args.mode == "sp":
        opt_joint_angles, opt_compliance, opt_target_pose, opt_palm_pose, opt_margin, opt_R, opt_t = grasp_optimizer.optimize(init_joint_angles,target_pose, compliance, friction_mu, gpis, verbose=True)
        opt_tip_pose = grasp_optimizer.forward_kinematics(opt_joint_angles, opt_palm_pose)
    elif args.mode == "fc":
        opt_tip_pose, opt_compliance, opt_target_pose, opt_palm_pose, opt_margin, opt_joint_angles = grasp_optimizer.optimize(init_joint_angles, target_pose, compliance, friction_mu, gpis, verbose=True)
    
    # Get feasible idx
    idx_list = []
    for i in range(opt_tip_pose.shape[0]):
        if opt_margin[i].min() > 0.0:
            idx_list.append(i)
        else:
            continue
    print("Feasible indices:",idx_list, "Feasible rate:", len(idx_list)/opt_tip_pose.shape[0])
    print("Optimal compliance:", opt_compliance)

    # Save grasps
    if args.exp_name is not None:
        if len(idx_list) > 0:
            np.save(f"data/contact_{args.exp_name}.npy", opt_tip_pose.cpu().detach().numpy()[idx_list])
            np.save(f"data/target_{args.exp_name}.npy", opt_target_pose.cpu().detach().numpy()[idx_list])
            np.save(f"data/wrist_{args.exp_name}.npy", opt_palm_pose.cpu().detach().numpy()[idx_list])
            np.save(f"data/compliance_{args.exp_name}.npy", opt_compliance.cpu().detach().numpy()[idx_list])
            if args.mode == "prob":
                np.save(f"data/joint_angle_{args.exp_name}.npy", opt_joint_angles.cpu().detach().numpy()[idx_list])
    else:
        data_dict = {
            "feasible_idx": np.array([idx_list]),
            "start_tip_pose": opt_tip_pose.cpu().detach().numpy(),
            "target_tip_pose": opt_target_pose.cpu().detach().numpy(),
            "palm_pose": opt_palm_pose.cpu().detach().numpy(),
            "compliance": opt_compliance.cpu().detach().numpy(),
            "input_pts": np.asarray(pcd.points),
            "opt_t": opt_t.cpu().detach().numpy(),
            "opt_R": opt_R.cpu().detach().numpy(),
            "input_path": args.npz_path,
        }
        save_path = os.path.join(os.path.dirname(args.npz_path), "sg_predictions.npz")
        print("Saving predictions to:", save_path)
        np.savez_compressed(save_path, data=data_dict)

    if args.vis is not None:
        if args.vis == "pb":
            # Visualize grasp in pybullet
            pcd.colors = o3d.utility.Vector3dVector(np.array([0.0, 0.0, 1.0] * len(pcd.points)).reshape(-1,3))
            grasp_vis = GraspVisualizer(robot_urdf, pcd)
            # After transformation
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            floor = o3d.geometry.TriangleMesh.create_box(width=0.5, height=0.5, depth=0.01).translate([-0.25,-0.25,-0.01])
            for i in idx_list:
                tips, targets, arrows = vis_grasp(opt_tip_pose[i], opt_target_pose[i])
                grasp_vis.visualize_grasp(joint_angles=opt_joint_angles[i].detach().cpu().numpy(), 
                                            wrist_pose=opt_palm_pose[i].detach().cpu().numpy(), 
                                            target_pose=opt_target_pose[i].detach().cpu().numpy())
                o3d.visualization.draw_geometries([pcd, *tips, *targets, *arrows])

        if args.vis == "o3d":
            for i in idx_list:
                print("Grasp:", i)
                v_utils.vis_results(
                    pcd, opt_tip_pose[i], opt_target_pose[i],
                    wrist_pose=opt_palm_pose[i].cpu().detach().numpy(),
                )