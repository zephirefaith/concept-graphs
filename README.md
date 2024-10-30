# ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and Planning

[**Project Page**](https://concept-graphs.github.io/) |
[**Paper**](https://concept-graphs.github.io/assets/pdf/2023-ConceptGraphs.pdf) |
[**ArXiv**](https://arxiv.org/abs/2309.16650) |
[**Video**](https://www.youtube.com/watch?v=mRhNkQwRYnc&feature=youtu.be&ab_channel=AliK)

[Qiao Gu](https://georgegu1997.github.io/)\*,
[Ali Kuwajerwala](https://www.alihkw.com/)\*,
[Sacha Morin](https://sachamorin.github.io/)\*,
[Krishna Murthy Jatavallabhula](https://krrish94.github.io/)\*,
[Bipasha Sen](https://bipashasen.github.io/),
[Aditya Agarwal](https://skymanaditya1.github.io/),
[Corban Rivera](https://www.jhuapl.edu/work/our-organization/research-and-exploratory-development/red-staff-directory/corban-rivera),
[William Paul](https://scholar.google.com/citations?user=92bmh84AAAAJ),
[Kirsty Ellis](https://mila.quebec/en/person/kirsty-ellis/),
[Rama Chellappa](https://engineering.jhu.edu/faculty/rama-chellappa/),
[Chuang Gan](https://people.csail.mit.edu/ganchuang/),
[Celso Miguel de Melo](https://celsodemelo.net/),
[Joshua B. Tenenbaum](http://web.mit.edu/cocosci/josh.html),
[Antonio Torralba](https://groups.csail.mit.edu/vision/torralbalab/),
[Florian Shkurti](http://www.cs.toronto.edu//~florian/),
[Liam Paull](http://liampaull.ca/)

![Splash Figure](./assets/splash-final.png)

## Using Concept-Graph repository with PARTNR Trajectories

First follow the instructions outlined [here](https://github.com/facebookresearch/partnr/tree/main/habitat_llm/concept_graphs) to setup concept-graph and get a trajectory in PARTNR scenes. 
Please follow the installation instructions provided in ioriginal CG repository](https://github.com/concept-graphs/concept-graphs) to install the library.

Once you have CG repository installed and trajectories stored follow the outlined steps to create a concept-graph.

### Export relevant environment variables

```bash
export GSA_PATH=/absolute/path/to/Grounded-Segment-Anything/
export HABITAT_DATA_ROOT=/absolute/path/to/trajectories/root/dir
export HABITAT_CONFIG_PATH=/absolute/path/to/concept-graphs/root/conceptgraphs/dataset/dataconfigs/habitat/habitat.yaml
export CLASS_SET=fix_set
```

All hyperparameters mentioned in this readme are the exact ones used for our experiments.

### Detect Objects and Create a 3D Object-centric Map based on Reassociation Across Views

First set the scene-name you want to create CG for: 

```bash
export SCENE_NAME=<scene-name>/main_agent/
```

Make sure to download YOLOv8x-worldv2 checkpoint and provide path to it in `conceptgraph/scripts/generate_gs_results.py` L528.
Run the following command to detect objects from closed-set:

```bash
python scripts/generate_gsa_results.py \
    --dataset_root $HABITAT_DATA_ROOT \
    --dataset_config $HABITAT_CONFIG_PATH \
    --scene_id $SCENE_NAME \
    --class_set $CLASS_SET \
    --box_threshold 0.2 \
    --text_threshold 0.2 \
    --stride 5 \
    --accumu_classes
```

Next, run the 3D association script to get 3D object-centric map of the scene: 
```bash
export THRESHOLD=0.9
python slam/cfslam_pipeline_batch.py \
dataset_root=$HABITAT_DATA_ROOT \
dataset_config=$HABITAT_CONFIG_PATH \
stride=5 \
scene_id=$SCENE_NAME \
spatial_sim_type=overlap \
mask_conf_threshold=0.25 \
match_method=sim_sum \
sim_threshold=$THRESHOLD \
dbscan_eps=0.1 \
gsa_variant=fix_set \
skip_bg=True \
max_bbox_area_ratio=0.75 \
save_suffix=overlap_maskconf0.25_simsum0.9_dbscan0.1_maxbboxarearatio0.75_merge0.8 \
merge_interval=20 \
merge_visual_sim_thresh=0.8 \
merge_text_sim_thresh=0.8
```

The above commands will save the mapping results in `$HABITAT_DATA_ROOT/$SCENE_NAME/pcd_saves`. 
It will create two pkl.gz files, where the one with _post suffix indicates results after some post processing, which we recommend using.

### Assign object-categories using CLIP and get room labels for each Entity

Set the full-name of pl;.gz file with _post suffix from above as `PKL_FILENAME`. 
```bash
export PKL_FILENAME=<full-name-of-post.pkl.gz-from-previous-run>
```

Run name refinement script which classifies each object against closed-vocab of object-names: 
```bash
python scripts/refine_node_names_using_clip.py \
  --input-file $HABITAT_DATA_ROOT/$SCENE_NAME/pcd_saves/$PKL_FILENAME \
  --threshold 0.15
```

Next run script which uses llama3.1:70b to assign room labels to each piece of furniture.


```bash
python scripts/get_room_labels.py \
  --input-file $HABITAT_DATA_ROOT/$SCENE_NAME/pcd_saves/$PKL_FILENAME
```

### Build the Scenegraph

```bash
python scripts/build_scenegraph_new_way.py \
--mode build-scenegraph \
--cachedir $HABITAT_DATA_ROOT/$SCENE_NAME \
--mapfile $HABITAT_DATA_ROOT/$SCENE_NAME/pcd_saves/$PKL_FILENAME \
--class_names_file $HABITAT_DATA_ROOT/$SCENE_NAME/gsa_classes_fix_set.json
```

### Key Differences

We do not use llava in our work to get open-vocabulary obejcts and furniture. Instead we use CLIP features from 3D reassociation step to classify each object against a category from the closed-set. 
We use llama3.1:70b to assign a room-label to each furniture to get the predicted layout of the house. Finally, since we do not use llava our scene-graph build step is just one step unlike original CG work.
