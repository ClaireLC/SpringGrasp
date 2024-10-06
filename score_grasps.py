import numpy as np
import argparse
import torch
from tqdm import tqdm
import os
import json

import spring_grasp_planner.func_metrics as f_metrics

device = "cpu"

def main(args):
    grasp_dict = np.load(args.grasp_path, allow_pickle=True)["data"].item() 

    # Open json file of hand-labeled grasp ids
    labeled_grasp_path = os.path.join(os.path.dirname(args.grasp_path), "hand_labeled_grasps.json")
    if os.path.exists(labeled_grasp_path):
        with open(labeled_grasp_path, "r") as f:
            labeled_grasp_info = json.load(f)
        grasps_to_eval = []
        for l, id_list in labeled_grasp_info.items():
            grasps_to_eval += id_list
        grasps_to_eval = np.array(grasps_to_eval)
    else:
        if "feasible_idx" in grasp_dict:
            feasible_idx = np.squeeze(grasp_dict["feasible_idx"])
        else:
            feasible_idx = None
        grasps_to_eval = feasible_idx
    
    if grasps_to_eval is None:
        print("No grasps to eval. Exiting.")
        return

    # Load affordance labels from input path 
    input_dict = np.load(grasp_dict["input_path"], allow_pickle=True)["data"].item()
    aff_labels = torch.tensor(input_dict["aff_labels"]).to(device)
    pts = torch.tensor(input_dict["pts_wf"]).to(device)

    # Load GPIS
    gpis_path = os.path.join(os.path.dirname(args.grasp_path), "gpis.pt")
    if not os.path.exists(gpis_path):
        gpis_path = os.path.join(os.path.dirname(os.path.dirname(args.grasp_path)), "gpis.pt")
    if not os.path.exists(gpis_path):
        raise ValueError("Cannot find GPIS at path", gpis_path)
    gpis = torch.load(gpis_path)
    print("Loaded GPIS")

    # TODO use contact positions, computed with pregrasp_coeffs (L 900 in optimizers.py)
    # Visualize this - see if they are roughly close to surface

    # Score a batch of grasps
    init_ftip_pos = torch.tensor(grasp_dict["start_tip_pose"][grasps_to_eval]).to(device).double()
    target_ftip_pos = torch.tensor(grasp_dict["target_tip_pose"][grasps_to_eval]).to(device).double()

    batch_size = init_ftip_pos.shape[0]

    """
    Compute Contact Points. Repeat target and pre-grasp positions based on
    number of sets of pre-grasp coeffs. Interleave pregrasp_coefficients accordingly.
    """
    # Create coefficients
    pregrasp_coefficients = torch.Tensor([[0.7, 0.7, 0.7, 0.7]]).to(device) # [1, 4]
    pregrasp_coeffs = pregrasp_coefficients.repeat_interleave(batch_size, dim=0)

    # Compute contact position p(t_0). Expected contact points.
    contact_pos = target_ftip_pos + pregrasp_coeffs.view(-1, 4, 1) * (init_ftip_pos - target_ftip_pos)

    pts_batched = pts.repeat(batch_size, 1, 1) # [B, N, 3]
    aff_labels_batched = aff_labels.repeat(batch_size, 1) # [B, N]

    print("Grasps to eval:", grasps_to_eval)

    """
    gpis: gpis
    pts: full point cloud (Tensor [batch_size, npoint, 3])
    ftip_pos: fingertip positions (Tensor [batch_size, 4, 3])
    aff_labels: per-point affordance labels (Tensor [batch_size, npoint])
    w_pos: weight for positive points (aff_label == 1)
    w_neg: weight for negative points (aff_label != 1)
    dp_thresh: threshold for checking dot product between ft_pos normal and pt normal
    dist_to_use: distance to use for computing cost ["gpis", "euclidean"]
    debug: if True, show debug visualizations in o3d
    """
    cost = f_metrics.contactgrasp_metric(gpis,
                                         pts_batched,
                                         contact_pos,
                                         # init_ftip_pos,
                                         aff_labels_batched,
                                         w_pos=args.w_pos,
                                         w_neg=args.w_neg,
                                         dp_thresh=args.dp_thresh,
                                         dist_to_use=args.dist,
                                         debug=args.debug,
                                         )
    print("Negative cost (high=better):")
    cost = cost.cpu().numpy()
    print(np.array_repr(-cost).replace("\n", ""))

    print("Sorted cost (low = better)")
    sorted_idxs = np.argsort(cost)
    sorted_cost = np.take_along_axis(cost, sorted_idxs, axis=0)
    sorted_ids = np.take_along_axis(grasps_to_eval, sorted_idxs, axis=0)
    print("Sorted IDs:", sorted_ids)
    print("Cost:", sorted_cost)


if __name__ == "__main__":

    """ For ann_pcd.npz -> Based on Predictions
    python score_grasps.py -wn 0 /juno/u/junhokim/code/zed_redis/pcd_data/real_plier/obj8/opt_fmn-none_wf-0p0_fcd-euclidean_fcwp-1p0_fcwn-1p0_fcdt-0p9_ffp-pregrasp_09-30-24_133948/sg_predictions.npz
    """

    """ For ann_gt_pcd.npz -> Based on Ground Truth
    python score_grasps.py -wn 0 /juno/u/junhokim/code/SpringGrasp/data/grasp/real_plier/obj1/sg_predictions.npz
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("grasp_path",
                        type=str,
                        help=".npz file with grasp optimization results",
                        )
    parser.add_argument("--w_pos", 
                        "-wp",
                        type=float,
                        default=1.0,
                        help="positive weights"
                        )
    parser.add_argument("--w_neg",
                        "-wn",
                        type=float,
                        default=1.0,
                        help="negative weights"
                        )
    parser.add_argument("--dp_thresh", 
                        type=float, 
                        default=0.9, 
                        help="if specified, threshold normal dp by this value"
                        )
    parser.add_argument("--dist", 
                        default="euclidean", 
                        choices=["gpis", "euclidean"], 
                        help="distance to use when computing score",
                        )
    parser.add_argument("--save", "-s", action="store_true")
    parser.add_argument("--debug", "-d", default="True", help="visualize")
    
    args = parser.parse_args()
    main(args)