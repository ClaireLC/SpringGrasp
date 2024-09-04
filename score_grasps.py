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
    # Visusalize this - see if they are roughly close to surface

    # Score a batch of grasps
    init_ftip_pos = torch.tensor(grasp_dict["start_tip_pose"][grasps_to_eval]).to(device).double()
    target_ftip_pos = torch.tensor(grasp_dict["target_tip_pose"][grasps_to_eval]).to(device).double()

    batch_size = init_ftip_pos.shape[0]

    # Compute contact points
    # Repeat target and pre-grasp positions based on number of sets of pre-grasp coeffs to try 
    # Interleave pregrasp_coefficients accordingly
    pregrasp_coefficients = torch.Tensor([[0.7, 0.7, 0.7, 0.7]]).to(device)
    #target_pose_extended = target_pose.repeat(len(pregrasp_coefficients),1,1) # [num_envs * len(self.pregrasp_coefficients), 4, 3]
    #pregrasp_tip_pose_extended = self.pregrasp_tip_pose.repeat(len(self.pregrasp_coefficients),1,1) #[e1,e2,e3,e4,e1,e2,e3,e4, ...]
    pregrasp_coeffs = pregrasp_coefficients.repeat_interleave(batch_size,dim=0)
    # Compute contact position p(t_0)
    contact_pos = target_ftip_pos + pregrasp_coeffs.view(-1, 4, 1) * (init_ftip_pos - target_ftip_pos)

    pts_batched = pts.repeat(batch_size, 1, 1) # [B, N, 3]
    aff_labels_batched = aff_labels.repeat(batch_size, 1) # [B, N]

    print("Grasps to eval:", grasps_to_eval)
    cost = f_metrics.contactgrasp_metric(
        gpis,
        pts_batched,
        contact_pos,
        #init_ftip_pos,
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "grasp_path",
        type=str,
        help="Path to .npz file with grasp optimization results",
    )
    parser.add_argument(
        "--w_pos", "-wp", type=float, default=1.0, help="positive weights"
    )
    parser.add_argument(
        "--w_neg", "-wn", type=float, default=1.0, help="negative weights"
    )
    parser.add_argument(
        "--dp_thresh", type=float, default=0.9, help="if specified, threshold normal dp by this value"
    )
    parser.add_argument(
        "--dist",
        default="euclidean",
        choices=["gpis", "euclidean"],
        help="distance to use when computing score",
    )
    parser.add_argument("--save", "-s", action="store_true")
    parser.add_argument("--debug", "-d", action="store_true")
    args = parser.parse_args()
    main(args)
