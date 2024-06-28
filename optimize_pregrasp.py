import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from gpis.gpis import GPIS
import torch
from utils import robot_configs
from utils.create_arrow import create_direct_arrow
import os
import wandb
from datetime import date, datetime

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
            
def get_run_name(conf, with_scene_name=False):
    today_date = date.today().strftime("%m-%d-%y")
    timestamp = datetime.now().time().strftime("%H%M%S")

    if args.npz_path is not None:
        obj_name = os.path.basename(os.path.dirname((os.path.dirname(args.npz_path))))
        pos_name = os.path.basename((os.path.dirname(args.npz_path)))
        scene_name = obj_name + "_" + pos_name
    else:
        scene_name = args.exp_name

    if with_scene_name:
        run_name = scene_name
    else:
        run_name = "opt"

    # Add conf params to name
    params_to_add = [
        "func_metric_name",
        "w_func",
        "func_contactgrasp_dist",
        "func_contactgrasp_w_pos",
        "func_contactgrasp_w_neg",
        "func_contactgrasp_dp_thresh",
        "func_finger_pts",
    ]
    
    conf_dict = vars(conf)
    for key in params_to_add:
        val = conf_dict[key]
        if isinstance(val, list):
            val_str = "-".join([str(i) for i in val]).replace(".", "p")
        elif type(val) == float:
            val_str = str(val).replace(".", "p")
        elif val is None:
            val_str = "none"
        else:
            val_str = str(val)

        if key in [
            "pretrained_pointnet_dir",
        ]:
            if val_str.lower() != "none":
                val_str = "true"

        # abbreviate key
        splits = key.split("_")
        short_key = ""

        for split in splits:
            short_key += split[0]
    
        run_name += f"_{short_key}-{val_str}"

    run_name += ("_" + today_date + "_" + timestamp)

    return run_name


def set_wandb_config(project_name, run_name, conf, wandb_entity="clairec"):
    """
    Set up wandb logging
    """

    wandb.require("service")
    # Load or save wandb info
    exp_dir = os.path.dirname(conf["npz_path"])
    wandb_info_path = os.path.join(exp_dir, "wandb_info.json")

    wandb_id = wandb.util.generate_id()
    wandb_info = {
        "run_name": run_name,
        "id": wandb_id,
        "project": project_name,
    }
    with open(wandb_info_path, "w") as f:
        json.dump(wandb_info, f, indent=4)

    # wandb init
    wandb.init(
        project=project_name,
        entity=wandb_entity,
        name=wandb_info["run_name"],
        id=wandb_info["id"],
        config=conf,
    )


def set_seeds():
    np.random.seed(0)
    torch.manual_seed(0)

if __name__ == "__main__":
    from argparse import ArgumentParser
    import json

    set_seeds()

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
    parser.add_argument("--vis", choices=["pb", "o3d"], help="Visualize mode. If not specified, do not show vis.")
    parser.add_argument("--vis_ic", action="store_true", help="Visualize initial conditions")
    parser.add_argument("--log", "-l", action="store_true", help="Log optimization to wandb")

    # Weights for loss terms - original SpringGrasp
    parser.add_argument("--w_sp", type=float, default=200, help="Weight for SpringGrasp cost")
    parser.add_argument("--w_dist", type=float, default=10000, help="Weight for contact pos distance")
    parser.add_argument("--w_uncer", type=float, default=20, help="Weight for uncertainty")
    parser.add_argument("--w_gain", type=float, default=0.5, help="Weight for regularizing gains")
    parser.add_argument("--w_tar", type=float, default=1000, help="Weight for target pos distance")
    parser.add_argument("--w_col", type=float, default=1.0, help="Weight for penalizing collisions")
    parser.add_argument("--w_reg", type=float, default=10.0, help="Weight for regularizing joint angles and fingertip postions")
    parser.add_argument("--w_force", type=float, default=200.0, help="Weight for regularizing contact forces")
    parser.add_argument("--w_pre_dist", type=float, default=50.0, help="Weight for pre-grasp finertip pos")
    parser.add_argument("--w_palm_dist", type=float, default=1.0, help="Weight for palm to obj distance")

    # Functional grasp params
    parser.add_argument("--w_func", type=float, default=0.0, help="Weight for functional grasp term")
    parser.add_argument("--func_metric_name", type=str, choices=["contactgrasp"], help="Name of functional grasp metric to use")
    parser.add_argument("--func_contactgrasp_dist", type=str, choices=["gpis", "euclidean"], default="euclidean")
    parser.add_argument("--func_contactgrasp_w_pos", type=float, default=1.0, help="Weight for positive points")
    parser.add_argument("--func_contactgrasp_w_neg", type=float, default=1.0, help="Weight for negative points")
    parser.add_argument("--func_contactgrasp_dp_thresh", type=float, default=0.9)
    parser.add_argument("--func_finger_pts", type=str, default="pregrasp", choices=["pregrasp", "contact", "target"], help="Finger points to use")


    args = parser.parse_args()

    if args.weight_config is not None:
        weight_config = json.load(open(f"weight_config/{args.weight_config}.json"))
    else:
        # Use default weights
        weight_config = {
            "w_sp": args.w_sp,
            "w_dist": args.w_dist,
            "w_tar": args.w_tar,
            "w_uncer": args.w_uncer,
            "w_gain": args.w_gain,
            "w_col": args.w_col,
            "w_reg": args.w_reg,
            "w_force": args.w_force,
            "w_pre_dist": args.w_pre_dist,
            "w_palm_dist": args.w_palm_dist,
            "w_func": args.w_func,
        }
    
    # Create run directory to log optimization results
    run_dir_name = get_run_name(args)
    if args.npz_path is not None:
        run_dir = os.path.join(os.path.dirname(args.npz_path), run_dir_name)
    elif args.pcd_file is not None:
        run_dir = os.path.dirname(args.pcd_file)
    else:
        run_dir = "output"
    if not os.path.exists(run_dir): os.makedirs(run_dir)

    # Set up wandb logging
    if args.log and args.npz_path is not None:
        run_name = get_run_name(args, with_scene_name=True)
        args_dict = vars(args)
        args_dict["run_dir"] = run_dir # Save run_dir to wandb log
        set_wandb_config("springgrasp", run_name, args_dict)
    
        # Save args_dict to run_dir
        conf_path = os.path.join(run_dir, "conf.json")
        with open(conf_path, "w") as f:
            json.dump(args_dict, f, indent=4)

    if args.pcd_file is not None:
        pcd = o3d.io.read_point_cloud(args.pcd_file)
        center = pcd.get_axis_aligned_bounding_box().get_center()
        #WRIST_OFFSET[:,0] += center[0]
        #WRIST_OFFSET[:,1] += center[1]
        #WRIST_OFFSET[:,2] += center[2]
        #init_wrist_poses = WRIST_OFFSET
        init_wrist_poses = init_cond.get_init_wrist_pose_from_pcd(pcd)
        input_pts = None
        aff_labels = None
        input_path = args.pcd_file
    elif args.npz_path is not None:
        input_dict = np.load(args.npz_path, allow_pickle=True)["data"].item() 
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(input_dict["pts_wf"])
        #init_wrist_poses = init_cond.get_default_wrist_pose(pcd)
        init_wrist_poses = init_cond.get_init_wrist_pose_from_pcd(pcd, viz=args.vis_ic, check_palm_ori=True)
        center = pcd.get_axis_aligned_bounding_box().get_center()
        
        input_pts = torch.tensor(input_dict["pts_wf"]).to(device).double()
        if "aff_labels" in input_dict:
            aff_labels = torch.tensor(input_dict["aff_labels"]).to(device).double()
        else:
            aff_labels = None
        input_path = args.npz_path
    else:
        pcd = o3d.io.read_point_cloud("data/obj_cropped.ply")
        center = pcd.get_axis_aligned_bounding_box().get_center()
        WRIST_OFFSET[:,0] += center[0]
        WRIST_OFFSET[:,1] += center[1]
        WRIST_OFFSET[:,2] += 2 * center[2]
        init_wrist_poses = WRIST_OFFSET
        input_pts = None
        aff_labels = None
        input_path = None
    
    # GPIS formulation - load or fit
    # TODO if using cuda, need to compute GPIS with cuda
    if args.npz_path is not None:
        gpis_save_path = os.path.join(os.path.dirname(args.npz_path), "gpis.pt")
    else:
        gpis_save_path = None
    bound = max(max(pcd.get_axis_aligned_bounding_box().get_extent()) / 2 + 0.01, 0.1) # minimum bound is 0.1
    if gpis_save_path is not None and os.path.exists(gpis_save_path):
        # Load
        print("Loading GPIS from", gpis_save_path)
        gpis = torch.load(gpis_save_path)
    else:
        # Fit
        print("Fitting GPIS...")
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
        if gpis_save_path is not None:
            torch.save(gpis, gpis_save_path)
            print("Saved GPIS to", gpis_save_path)
    if args.vis_gpis:
        print("Visualizing GPIS...")
        test_mean, test_var, test_normal, lb, ub = gpis.get_visualization_data([-bound+center[0],-bound+center[1],-bound+center[2]],
                                                                            [bound+center[0],bound+center[1],bound+center[2]],steps=100)
        plt.imshow(test_mean[:,:,50], cmap="seismic", vmax=bound, vmin=-bound)
        plt.show()
        vis_points, vis_normals, vis_var = gpis.topcd(
            test_mean,
            test_normal,
            [-bound+center[0],-bound+center[1],-bound+center[2]],
            [bound+center[0],bound+center[1],bound+center[2]],
            test_var=test_var,
            steps=100,
        )
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
            if args.npz_path is not None:
                save_path = os.path.join(os.path.dirname(args.npz_path), "sg_gpis.npz")
                print("Saving GPIS results to:", save_path)
                np.savez_compressed(save_path, data=gpis_dict)
                save_path = os.path.join(os.path.dirname(args.npz_path), "sg_gpis.ply")
                print("Saving GPIS viz pcd to:", save_path)
                o3d.io.write_point_cloud(save_path, fitted_pcd)  
        quit()

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
        grasp_optimizer = optimizers[args.mode](
            robot_urdf,
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
            gravity=False
        )
    elif args.mode == "sp":
        grasp_optimizer = optimizers[args.mode](
            robot_urdf,
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
            mass=args.mass,
            com=center[:3],
            gravity=False,
            weight_config=weight_config,
            conf=args,
        )

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
        save_path = os.path.join(run_dir, "sg_init_cond.npz")
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
        opt_joint_angles, opt_compliance, opt_target_pose, opt_palm_pose, opt_margin, opt_R, opt_t = grasp_optimizer.optimize(
            init_joint_angles,
            target_pose,
            compliance,
            friction_mu,
            gpis,
            pts=input_pts,
            aff_labels=aff_labels,
            verbose=True,
        )
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
            "input_path": input_path,
        }
        save_path = os.path.join(run_dir, "sg_predictions.npz")
        print("Saving predictions to:", save_path)
        np.savez_compressed(save_path, data=data_dict)

    if args.vis is not None:
        if args.vis == "pb":
            from utils.pb_grasp_visualizer import GraspVisualizer
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