import argparse
import json
import os
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from gradslam.slam.pointfusion import PointFusion
from gradslam.structures.pointclouds import Pointclouds
from gradslam.structures.rgbdimages import RGBDImages
from tqdm import trange

from conceptgraph.dataset.datasets_common import get_dataset


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        required=True,
        help="This path may need to be changed depending on where you run this script. ",
    )
    parser.add_argument("--scene_id", type=str, default="train_3")
    parser.add_argument("--image_height", type=int, default=480)
    parser.add_argument("--image_width", type=int, default=640)

    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--stride", type=int, default=1)

    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--save_pcd", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument(
        "--load_semseg",
        action="store_true",
        help="Load GT semantic segmentation and run fusion on them. ",
    )

    return parser


def main(args: argparse.Namespace):
    if args.load_semseg:
        load_embeddings = True
        embedding_dir = "embed_semseg"
        semseg_classes = json.load(
            open(args.dataset_root / args.scene_id /
                 "embed_semseg_classes.json", "r")
        )
        embedding_dim = len(semseg_classes)
    else:
        load_embeddings = False
        embedding_dir = "embeddings"
        embedding_dim = 512

    dataset = get_dataset(
        dataconfig=args.dataset_config,
        basedir=args.dataset_root,
        sequence=args.scene_id,
        desired_height=args.image_height,
        desired_width=args.image_width,
        start=args.start,
        end=args.end,
        stride=args.stride,
        load_embeddings=load_embeddings,
        embedding_dir=embedding_dir,
        embedding_dim=embedding_dim,
        relative_pose=False,
    )
    slam_mode = "gt"

    slam = PointFusion(odom=slam_mode, dsratio=1, device=args.device)

    frame_cur, frame_prev = None, None
    pointclouds = Pointclouds(device=args.device)
    dir_to_save_map = os.path.join(
        args.dataset_root, args.scene_id, "rgb_cloud")
    pose_list = []

    for idx in trange(len(dataset)):
        if load_embeddings:
            _color, _depth, intrinsics, _pose, _embedding = dataset[idx]
            _embedding = _embedding.unsqueeze(0).half()
            _confidence = torch.ones_like(_embedding)
        else:
            _color, _depth, intrinsics, _pose = dataset[idx]
            _embedding = None
            _confidence = None

        pose_np = _pose.cpu().numpy()

        _pose = torch.from_numpy(pose_np).to(_pose.device).to(_pose.dtype)

        frame_cur = RGBDImages(
            _color.unsqueeze(0).unsqueeze(0),
            _depth.unsqueeze(0).unsqueeze(0),
            intrinsics.unsqueeze(0).unsqueeze(0),
            _pose.unsqueeze(0).unsqueeze(0),
            embeddings=_embedding,
            confidence_image=_confidence,
        )
        if slam_mode != "gt":
            print(f"Current pose: {frame_cur.poses}")

        pointclouds, new_poses = slam.step(pointclouds, frame_cur, frame_prev)
        # temp_pcd = pointclouds.open3d(0)
        # o3d.io.write_point_cloud(
        #     os.path.join(dir_to_save_map, f"pointcloud_{idx}.pcd"), temp_pcd
        # )  # Saving as PCD
        if slam_mode != "gt":
            frame_prev = frame_cur  # Keep it None when we use the gt odom
            frame_prev.poses = new_poses
            print(f"Inferred pose: {new_poses}")
            pose_list.append(new_poses.detach().cpu().numpy())
        torch.cuda.empty_cache()
        # if idx == 0:
        #     exit()

    print(f"Saving the map to {dir_to_save_map}")
    os.makedirs(dir_to_save_map, exist_ok=True)
    pointclouds.save_to_h5(dir_to_save_map)

    if slam_mode != "gt":
        print("Saving poses to ./new_poses/")
        os.makedirs("./new_poses/", exist_ok=True)
        for pose_idx, pose in enumerate(pose_list):
            np.save(f"./new_poses/{pose_idx}.npy", pose)

    # Set the filename for the PCD file
    pcd_file_path = os.path.join(dir_to_save_map, "pointcloud.pcd")
    pcd = pointclouds.open3d(0)

    if args.save_pcd:
        o3d.io.write_point_cloud(pcd_file_path, pcd)  # Saving as PCD

    if args.visualize:
        o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    args = get_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    main(args)
