python scenegraph/build_scenegraph_cfslam.py \
   --mode extract-node-captions \
   --cachedir /private/home/priparashar/$1/sg_cache \
   --mapfile /private/home/priparashar/$1/pcd_saves/$2 \
   --class_names_file /private/home/priparashar/$1/gsa_classes_ram_withbg_allclasses.json

python scenegraph/build_scenegraph_cfslam.py \
   --mode refine-node-captions \
   --cachedir /private/home/priparashar/$1/sg_cache \
   --mapfile /private/home/priparashar/$1/pcd_saves/$2 \
   --class_names_file /private/home/priparashar/$1/gsa_classes_ram_withbg_allclasses.json

python scenegraph/build_scenegraph_cfslam.py \
   --mode build-scenegraph \
   --cachedir /private/home/priparashar/$1/sg_cache \
   --mapfile /private/home/priparashar/$1/pcd_saves/$2 \
   --class_names_file /private/home/priparashar/$1/gsa_classes_ram_withbg_allclasses.json

python scenegraph/build_scenegraph_cfslam.py \
    --mode generate-scenegraph-json \
    --cachedir /private/home/priparashar/$1/sg_cache \
    --mapfile /private/home/priparashar/$1/pcd_saves/$2 \
    --class_names_file /private/home/priparashar/$1/gsa_classes_ram_withbg_allclasses.json
