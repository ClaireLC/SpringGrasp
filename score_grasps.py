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

    if "feasible_idx" in grasp_dict:
        feasible_idx = grasp_dict["feasible_idx"]
    else:
        feasible_idx = None

    # Load affordance labels from input path 
    input_dict = np.load(grasp_dict["input_path"], allow_pickle=True)["data"].item()
    aff_labels = torch.tensor(input_dict["aff_labels"]).to(device)
    pts = torch.tensor(input_dict["pts_wf"]).to(device)

    # Load GPIS
    gpis_path = os.path.join(os.path.dirname(args.grasp_path), "gpis.pt")
    gpis = torch.load(gpis_path)
    print("Loaded GPIS")

    # Score a batch of grasps
    init_ftip_pos = torch.tensor(grasp_dict["start_tip_pose"][grasps_to_eval]).to(device).double()
    batch_size = init_ftip_pos.shape[0]
    pts_batched = pts.repeat(batch_size, 1, 1) # [B, N, 3]
    aff_labels_batched = aff_labels.repeat(batch_size, 1) # [B, N]
    cost = f_metrics.contactgrasp_metric(
        gpis,
        pts_batched,
        init_ftip_pos,
        aff_labels_batched,
        w_pos=args.w_pos,
        w_neg=args.w_neg,
        dp_thresh=args.dp_thresh,
        dist_to_use=args.dist,
        debug=args.debug,
    )
    print(-cost)
    quit()

    ## Score one grasp at a time (unbatched)
    ## Iterate through grasps in grasp_path.npz
    #cost_list = []
    #for grasp_i in grasps_to_eval:
    #    print("Grasp", grasp_i)

    #    init_ftip_pos = grasp_dict["start_tip_pose"][grasp_i]
    #    target_ftip_pos = grasp_dict["target_tip_pose"][grasp_i]
    #    palm_pose = grasp_dict["palm_pose"][grasp_i]

    #    cost = f_metrics.contactgrasp_metric(
    #        gpis,
    #        pts,
    #        init_ftip_pos,
    #        aff_labels,
    #        w_pos=args.w_pos,
    #        w_neg=args.w_neg,
    #        dp_thresh=args.dp_thresh,
    #        dist_to_use=args.dist,
    #        debug=args.debug,
    #    )
    #    print("cost", cost)
    #    cost_list.append(-1 * cost) # Negate cost for logging/plotting
    #print(cost_list)
    

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
        "--dp_thresh", type=float, help="if specified, threshold normal dp by this value"
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
