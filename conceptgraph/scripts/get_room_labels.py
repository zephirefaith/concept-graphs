import gzip
import json
import pickle

import click
import numpy as np

from conceptgraph.slam.slam_classes import MapObjectList
from conceptgraph.utils.llm_local import LLMLocal

ROOM_LABEL_PROMPT: str = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an expert on house layouts. You will be given an input which will describe QUERY_OBJECT. This object will be described by its name and the 10 pieces of furniture closest to it. You will assign a ROOM_NAME to this object. The input will be a JSON with fields "QUERY_OBJECT_NAME" and "CLOSEST_OBJECTS".  Your output should also be a JSON consisting of key "ROOM_LABEL". You should only output the JSON and nothing else.

You should only assign one of the following labels:
1. bedroom
2. living room
3. kitchen
4. dining room
5. hallway
6. bathroom
7. unknown: only when none of the above strings describe the object<|eot_id|>"""


def get_query_prompt(input_json_str):
    prompt = f"\n<|start_header_d|>user<|end_header_id|>{input_json_str}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
    return prompt


@click.command()
@click.option("--input-file", default="map.json")
@click.option("--llm-mode", default="ollama", type=click.Choice(["ollama", "rlm"]))
def main(input_file, llm_mode):
    if llm_mode == "rlm":
        raise NotImplementedError("rlm mode not implemented")
    llm = LLMLocal(model_name="llama3.1:70b")

    # load the scene-map
    map_output = None
    with gzip.open(input_file, "rb") as f:
        map_output = pickle.load(f)
    objects_scene_map = MapObjectList()
    objects_scene_map.load_serializable(map_output["objects"])
    objects_raw = map_output["objects"]

    object_names, object_poses, object_indx = [], [], []
    for obj_idx, obj in enumerate(objects_scene_map):
        if obj["assigned_class_name"] != "INVALID":
            object_names.append(obj["assigned_class_name"])
            object_poses.append(np.array(obj["bbox"].center))
            object_indx.append(obj_idx)
    object_poses = np.array(object_poses)
    object_names = np.array(object_names)

    # for each object, find the closest 10 objects to the pose
    # fill the LLM's prompt with the closest 5 object's name and ask
    # for room assignment
    for current_obj_idx, current_object, current_pose in zip(
        object_indx, object_names, object_poses
    ):
        # get closest 5 objects to current pose
        closest = object_names[
            np.argsort(np.linalg.norm(current_pose - object_poses, axis=1))[:10]
        ]
        input_dict = {
            "QUERY_OBJECT_NAME": current_object,
            "CLOSEST_OBJECTS": closest,
        }
        print(f"Input: {input_dict}")
        full_prompt = ROOM_LABEL_PROMPT + get_query_prompt(input_dict)
        raw_response = llm.prompt(full_prompt)
        response = raw_response["message"]
        print(response)
        try:
            response = json.loads(response)["ROOM_LABEL"]
            objects_raw[current_obj_idx]["room_region"] = response
        except Exception as e:
            print(f"Error in parsing response: {e}")

    map_output["objects"] = objects_raw
    # update the json file with room assignment
    with gzip.open(input_file, "wb") as f:
        pickle.dump(map_output, f)


main()
