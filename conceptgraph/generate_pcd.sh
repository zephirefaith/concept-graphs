CUDA_LAUNCH_BLOCKING=1 python scripts/run_slam_rgb.py \
    --dataset_root /private/home/priparashar \
    --dataset_config /private/home/priparashar/SIRo/concept-graphs/conceptgraph/dataset/dataconfigs/habitat/habitat.yaml \
    --scene_id $1 \
    --save_pcd
