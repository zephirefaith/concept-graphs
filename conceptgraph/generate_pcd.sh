CUDA_LAUNCH_BLOCKING=1 python scripts/run_slam_rgb.py \
    --dataset_root /private/home/priparashar/SIRo/habitat-llm/data \
    --dataset_config /private/home/priparashar/SIRo/concept-graphs/conceptgraph/dataset/dataconfigs/habitat/habitat.yaml \
    --scene_id $1 \
    --stride 10 \
    --save_pcd
