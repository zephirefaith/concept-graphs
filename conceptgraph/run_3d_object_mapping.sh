THRESHOLD=1.2
python slam/cfslam_pipeline_batch.py \
    dataset_root=/private/home/priparashar \
    dataset_config=/private/home/priparashar/SIRo/concept-graphs/conceptgraph/dataset/dataconfigs/habitat/habitat.yaml \
    stride=5 \
    scene_id=$1 \
    spatial_sim_type=overlap \
    mask_conf_threshold=0.25 \
    match_method=sim_sum \
    sim_threshold=${THRESHOLD} \
    dbscan_eps=0.1 \
    gsa_variant=ram_withbg_allclasses \
    skip_bg=False \
    max_bbox_area_ratio=0.5 \
    save_suffix=overlap_maskconf0.25_simsum${THRESHOLD}_dbscan.1
