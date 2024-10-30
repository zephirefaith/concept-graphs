"""
Build a scene graph from the segment-based map, CLIP based class assignment and
LLAMA3.1:70B based room_region assignment
"""

import gc
import gzip
import json
import os
import pickle as pkl
import time
from dataclasses import dataclass
from pathlib import Path
from textwrap import wrap
from types import SimpleNamespace
from typing import List, Literal, Optional, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import openai
import rich
import torch
import tyro
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from tqdm import tqdm, trange
from transformers import logging as hf_logging

from conceptgraph.scenegraph.build_scenegraph_cfslam import ProgramArgs
from conceptgraph.utils.general_utils import prjson
from conceptgraph.utils.llm_local import LLMLocal

try:
    import rlm
    from rlm.llm import RemoteLanguageModel
except:
    print("RLM module not found")

from conceptgraph.scenegraph.build_scenegraph_cfslam import load_scene_map


def build_scenegraph(args):
    from conceptgraph.slam.slam_classes import MapObjectList
    from conceptgraph.slam.utils import compute_overlap_matrix

    # Load the scene map
    scene_map = MapObjectList()
    load_scene_map(args, scene_map)

    # load the llm (if required)
    rlm_model: Optional[RemoteLanguageModel] = None
    ollama_model: Optional[LLMLocal] = None
    if args.llm == "rlm":
        rlm_model = RemoteLanguageModel(f"http://{args.rlm_ip}:{args.rlm_port}")

    if args.llm == "ollama":
        ollama_model = LLMLocal(model_name="llama3.1:70b")

    responses = []
    indices_to_remove = []  # indices to remove if the json file does not exist
    for obj_idx, current_object in enumerate(scene_map):
        if current_object["assigned_class_name"] != "INVALID":
            responses.append(current_object)
        else:
            # Remove segments that correspond to "invalid" tags
            indices_to_remove.append(obj_idx)

    # Also remove segments that do not have a minimum number of observations
    indices_to_remove = set(indices_to_remove)
    for obj_idx in range(len(scene_map)):
        conf = scene_map[obj_idx]["conf"]
        # Remove objects with less than args.min_views_per_object observations
        if len(conf) < args.min_views_per_object:
            indices_to_remove.add(obj_idx)
    indices_to_remove = list(indices_to_remove)

    # List of tags in original scene map that are in the pruned scene map
    segment_ids_to_retain = [
        i for i in range(len(scene_map)) if i not in indices_to_remove
    ]
    with open(Path(args.cachedir) / "cfslam_scenegraph_invalid_indices.pkl", "wb") as f:
        pkl.dump(indices_to_remove, f)
    print(f"Removed {len(indices_to_remove)} segments which were invalid")

    # Filtering responses based on segment_ids_to_retain
    responses = [
        resp for idx, resp in enumerate(responses) if idx in segment_ids_to_retain
    ]

    # Assuming each response dictionary contains an 'object_tag' key for the object tag.
    # Extract filtered object tags based on filtered_responses
    object_tags = [resp["assigned_class_name"] for resp in responses]
    category_tags = ["furniture" for _resp in responses]
    orginal_class_names = [resp["class_name"] for resp in responses]

    pruned_scene_map = []
    for _idx, segmentidx in enumerate(segment_ids_to_retain):
        pruned_scene_map.append(scene_map[segmentidx])

    scene_map = MapObjectList(pruned_scene_map)
    del pruned_scene_map
    gc.collect()
    num_segments = len(scene_map)

    # Save the pruned scene map (create the directory if needed)
    if not (Path(args.cachedir) / "map").exists():
        (Path(args.cachedir) / "map").mkdir(parents=True, exist_ok=True)
    with gzip.open(
        Path(args.cachedir) / "map" / "scene_map_cfslam_pruned.pkl.gz", "wb"
    ) as f:
        pkl.dump(scene_map.to_serializable(), f)

    print("Computing bounding box overlaps...")
    bbox_overlaps = compute_overlap_matrix(args, scene_map)

    # Construct a weighted adjacency matrix based on similarity scores
    weights = []
    rows = []
    cols = []
    for i in range(num_segments):
        for j in range(i + 1, num_segments):
            if i == j:
                continue
            if bbox_overlaps[i, j] > 0.01:
                weights.append(bbox_overlaps[i, j])
                rows.append(i)
                cols.append(j)
                weights.append(bbox_overlaps[i, j])
                rows.append(j)
                cols.append(i)

    adjacency_matrix = csr_matrix(
        (weights, (rows, cols)), shape=(num_segments, num_segments)
    )

    # Find the minimum spanning tree of the weighted adjacency matrix
    mst = minimum_spanning_tree(adjacency_matrix)

    # Find connected components in the minimum spanning tree
    _, labels = connected_components(mst)

    components = []
    _total = 0
    if len(labels) != 0:
        for label in range(labels.max() + 1):
            indices = np.where(labels == label)[0]
            _total += len(indices.tolist())
            components.append(indices.tolist())

    with open(Path(args.cachedir) / "cfslam_scenegraph_components.pkl", "wb") as f:
        pkl.dump(components, f)

    # Initialize a list to store the minimum spanning trees of connected components
    minimum_spanning_trees = []
    relations = []
    if len(labels) != 0:
        # Iterate over each connected component
        for label in range(labels.max() + 1):
            component_indices = np.where(labels == label)[0]
            # Extract the subgraph for the connected component
            subgraph = adjacency_matrix[component_indices][:, component_indices]
            # Find the minimum spanning tree of the connected component subgraph
            _mst = minimum_spanning_tree(subgraph)
            # Add the minimum spanning tree to the list
            minimum_spanning_trees.append(_mst)

        TIMEOUT = 25  # timeout in seconds
        invalid_object = {
            "id": -1,
            "bbox_center": [0.0, 0.0, 0.0],
            "bbox_extent": [0.0, 0.0, 0.0],
            "object_tag": "INVALID",
            "category_tag": "INVALID",
            "original_class_name": [],
        }
        invalid_relation = {"object_relation": "FAIL"}

        # print(components) # TODO: Remove later, use for debug now
        if not (Path(args.cachedir) / "cfslam_object_relations.json").exists():
            relation_queries = []
            for componentidx, component in enumerate(components):
                if len(component) <= 1:
                    # even for components with only one object, we should store the
                    # object with an INVALID object connected via INVALID relation
                    current_object = scene_map[component[0]]
                    obj_bbox = current_object["bbox"]
                    output_dict = {
                        "object1": {
                            "id": component[0],
                            "bbox_extent": np.round(obj_bbox.extent, 1).tolist(),
                            "bbox_center": np.round(obj_bbox.center, 1).tolist(),
                            "object_tag": current_object["assigned_class_name"],
                            "category_tag": "furniture",
                            "original_class_name": current_object["class_name"],
                            "room_region": current_object["room_region"],
                        },
                        "object2": invalid_object,
                    }
                    output_dict.update(invalid_relation)
                    relations.append(output_dict)
                    continue
                for u, v in zip(
                    minimum_spanning_trees[componentidx].nonzero()[0],
                    minimum_spanning_trees[componentidx].nonzero()[1],
                ):
                    segmentidx1 = component[u]
                    segmentidx2 = component[v]
                    object1 = scene_map[segmentidx1]
                    object2 = scene_map[segmentidx2]
                    _bbox1 = scene_map[segmentidx1]["bbox"]
                    _bbox2 = scene_map[segmentidx2]["bbox"]

                    input_dict = {
                        "object1": {
                            "id": segmentidx1,
                            "bbox_extent": np.round(_bbox1.extent, 1)[
                                [0, 2, 1]
                            ].tolist(),
                            "bbox_center": np.round(_bbox1.center, 1)[
                                [0, 2, 1]
                            ].tolist(),
                            # "bbox_extent": np.round(_bbox1.extent, 1).tolist(),
                            # "bbox_center": np.round(_bbox1.center, 1).tolist(),
                            "object_tag": object1["assigned_class_name"],
                            "category_tag": "furniture",
                            "original_class_name": object1["class_name"],
                            "fix_bbox": True,
                            "room_region": object1["room_region"],
                        },
                        "object2": {
                            "id": segmentidx2,
                            "bbox_extent": np.round(_bbox2.extent, 1)[
                                [0, 2, 1]
                            ].tolist(),
                            "bbox_center": np.round(_bbox2.center, 1)[
                                [0, 2, 1]
                            ].tolist(),
                            # "bbox_extent": np.round(_bbox2.extent, 1).tolist(),
                            # "bbox_center": np.round(_bbox2.center, 1).tolist(),
                            "object_tag": object2["assigned_class_name"],
                            "category_tag": "furniture",
                            "original_class_name": object2["class_name"],
                            "room_region": object2["room_region"],
                            "fix_bbox": True,
                        },
                    }
                    print(
                        f"Processing: {input_dict['object1']['id']}_{input_dict['object1']['object_tag']}, {input_dict['object2']['id']}_{input_dict['object2']['object_tag']}"
                    )

                    relation_queries.append(input_dict)

                    input_json_str = json.dumps(input_dict)

                    start_time = time.time()
                    if args.llm == "openai":
                        # Default prompt
                        DEFAULT_PROMPT = """
The input is a list of JSONs describing two objects "object1" and "object2". You need to produce a JSON
string (and nothing else), with two keys: "object_relation", and "reason".

Each of the JSON fields "object1" and "object2" will have the following fields:
1. bbox_extent: the 3D bounding box extents of the object
2. bbox_center: the 3D bounding box center of the object
3. object_tag: an extremely brief description of the object

The bounding-boxes for given objects follow XYZ coordinate convention, with Z being the
height off of the ground.

Produce an "object_relation" field that best describes the relationship between the two objects. The
"object_relation" field must be one of the following (verbatim):
1. "a on b": if object a is an object commonly placed on top of object b
2. "b on a": if object b is an object commonly placed on top of object a
3. "a in b": if object a is an object commonly placed inside object b
4. "b in a": if object b is an object commonly placed inside object a
5. "none of these": if none of the above best describe the relationship between the two objects

Before producing the "object_relation" field, produce a "reason" field that explains why
the chosen "object_relation" field is the best.
"""
                        response = openai.ChatCompletion.create(
                            # model="gpt-3.5-turbo",
                            model="gpt-4",
                            messages=[
                                {
                                    "role": "user",
                                    "content": DEFAULT_PROMPT + "\n\n" + input_json_str,
                                }
                            ],
                            timeout=TIMEOUT,  # Timeout in seconds
                        )
                    elif args.llm == "rlm" or args.llm == "ollama":
                        DEFAULT_PROMPT = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>The input is a list of JSONs
describing two objects "object1" and "object2". You need to produce a JSON string (and
nothing else), with three keys: "object_relation", and "reason".

Each of the JSON fields "object1" and "object2" will have the following fields:
1. bbox_extent: the 3D bounding box extents of the object
2. bbox_center: the 3D bounding box center of the object
3. object_tag: an extremely brief description of the object
4. category_tag: categorization whether the object is a furniture or an object. 
5. orginal_class_name: the original class of the object

Produce an "object_relation" field that best describes the relationship between the two objects. The "object_relation" field must be one of the following (verbatim):
1. "a on b": if object a is an object commonly placed on top of object b
2. "b on a": if object b is an object commonly placed on top of object a
3. "a in b": if object a is an object commonly placed inside object b
4. "b in a": if object b is an object commonly placed inside object a
5. "a next to b": if object a is an object commonly placed next to object b
6. "none of these": if none of the above best describe the relationship between the two objects

Before producing the "object_relation" field, produce a "reason" field that explains why
the chosen "object_relation" field is the best. Think step-by-step. Make sure
"object_relation" field adhere to given list above. Only output the JSON and nothing else.<|eot_id|>
"""
                        full_prompt = (
                            DEFAULT_PROMPT
                            + f"\n<|start_header_id|>user<|end_header_id|>{input_json_str}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
                        )

                        if args.llm == "rlm":
                            raw_response = rlm_model.generate(
                                full_prompt, 500, temperature=0.0
                            )
                            if "assistant" in raw_response["generation"]:
                                response = raw_response["generation"].split(
                                    "assistant<|end_header_id|>"
                                )[1]
                            response = response.split("<|eot_id|>")[0]
                        elif args.llm == "ollama":
                            raw_response = ollama_model.prompt(full_prompt)
                            response = raw_response["message"]

                    elapsed_time = time.time() - start_time
                    output_dict = input_dict
                    if args.llm == "openai":
                        if elapsed_time > TIMEOUT:
                            print("Timed out exceeded!")
                            output_dict["object_relation"] = "FAIL"
                            continue
                        else:
                            try:
                                # Attempt to parse the output as a JSON
                                chat_output_json = json.loads(
                                    response["choices"][0]["message"]["content"]
                                )
                                # If the output is a valid JSON, then add it to the output dictionary
                                output_dict["object_relation"] = chat_output_json[
                                    "object_relation"
                                ]
                                output_dict["reason"] = chat_output_json["reason"]
                            except:
                                output_dict["object_relation"] = "FAIL"
                                output_dict["reason"] = "FAIL"
                    elif args.llm == "rlm" or args.llm == "ollama":
                        try:
                            # Attempt to parse the output as a JSON
                            chat_output_json = json.loads(response.strip())
                            # If the output is a valid JSON, then add it to the output dictionary
                            output_dict["object_relation"] = chat_output_json[
                                "object_relation"
                            ]
                            output_dict["reason"] = chat_output_json["reason"]
                            # output_dict["room_region"] = chat_output_json["room_region"]
                        except json.decoder.JSONDecodeError as e:
                            print("----FAILED TO OPEN LLM RESPONSE----")
                            output_dict["object_relation"] = "FAIL"
                            output_dict["reason"] = "FAIL"
                            # output_dict["room_region"] = "FAIL"
                            output_dict["summary"] = (
                                f"JSON output could not be decoded: {e}"
                            )
                            output_dict["raw_response"] = (
                                raw_response["generation"]
                                if args.llm == "rlm"
                                else raw_response
                            )
                    prjson(output_dict)

                    relations.append(output_dict)

            # Save the query JSON to file
            print("Saving query JSON to file...")
            with open(
                Path(args.cachedir) / "cfslam_object_relation_queries.json", "w"
            ) as f:
                json.dump(relation_queries, f, indent=4)

            # Saving the output
            print("Saving object relations to file...")
            with open(Path(args.cachedir) / "cfslam_object_relations.json", "w") as f:
                json.dump(relations, f, indent=4)
        else:
            print(
                f"Found file {Path(args.cachedir) / 'cfslam_object_relations.json'}, will load its json content"
            )
            relations = json.load(
                open(Path(args.cachedir) / "cfslam_object_relations.json", "r")
            )

    scenegraph_edges = []

    _idx = 0
    for componentidx, component in enumerate(components):
        if len(component) <= 1:
            continue
        for u, v in zip(
            minimum_spanning_trees[componentidx].nonzero()[0],
            minimum_spanning_trees[componentidx].nonzero()[1],
        ):
            segmentidx1 = component[u]
            segmentidx2 = component[v]
            # print(f"{segmentidx1}, {segmentidx2}, {relations[_idx]['object_relation']}")
            if relations[_idx]["object_relation"] != "none of these":
                scenegraph_edges.append(
                    (segmentidx1, segmentidx2, relations[_idx]["object_relation"])
                )
            _idx += 1
    print(
        f"Created 3D scenegraph with {num_segments} nodes and {len(scenegraph_edges)} edges"
    )

    with open(Path(args.cachedir) / "cfslam_scenegraph_edges.pkl", "wb") as f:
        pkl.dump(scenegraph_edges, f)


def main():
    # Process command-line args (if any)
    args = tyro.cli(ProgramArgs)

    # print using masking option
    print(f"args.masking_option: {args.masking_option}")

    if args.mode == "build-scenegraph":
        build_scenegraph(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
