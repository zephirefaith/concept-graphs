export CLASS_SET=ram
export GSA_PATH=/private/home/priparashar/SIRo/Grounded-Segment-Anything
python scripts/generate_gsa_results.py \
    --dataset_root /private/home/priparashar/ \
    --dataset_config /private/home/priparashar/SIRo/concept-graphs/conceptgraph/dataset/dataconfigs/habitat/habitat.yaml \
    --scene_id $1 \
    --class_set $CLASS_SET \
    --box_threshold 0.2 \
    --text_threshold 0.2 \
    --add_bg_classes \
    --accumu_classes \
    --exp_suffix withbg_allclasses \
    --stride 5
