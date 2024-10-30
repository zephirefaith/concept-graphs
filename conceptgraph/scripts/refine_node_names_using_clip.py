import gzip
import pickle

import click
import open_clip
import torch
from tqdm import tqdm


@click.command()
@click.option("--input-file", help="Input file containing the node names to refine")
@click.option("--threshold", default=0.20, help="Threshold for similarity")
@click.option("--score-ratio-threshold", default=1.5, help="Threshold for score ratio")
def main(input_file, threshold, score_ratio_threshold):
    results = None
    with gzip.open(input_file, "rb") as f:
        results = pickle.load(f)
    all_objects = results["objects"]
    class_names = results["class_names"]
    print(f"All class-names: {class_names}")

    # initialize the CLIP model
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14-quickgelu", "metaclip_fullcc"
    )
    clip_model = clip_model.to("cuda")
    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14-quickgelu")
    torch.set_grad_enabled(False)
    total_valid_objects = 0
    class_names_ft = clip_model.encode_text(clip_tokenizer(class_names).to("cuda"))
    class_names_ft /= class_names_ft.norm(dim=-1, keepdim=True)

    for current in tqdm(all_objects):
        object_clip_ft = torch.from_numpy(current["clip_ft"]).to("cuda")

        text_probs_based_on_vis_ft = (
            100.0 * object_clip_ft @ class_names_ft.T
        ).softmax(dim=-1)
        max_score = text_probs_based_on_vis_ft.max(-1)[0].cpu().numpy()
        print(f"Max score: {max_score}")
        if max_score > threshold:
            class_name_idx = text_probs_based_on_vis_ft.argmax(-1).cpu().numpy()
            current["assigned_class_name"] = class_names[class_name_idx]
            total_valid_objects += 1
        else:
            current["assigned_class_name"] = "INVALID"
        print(f"Assigned class name based on vis-ft: {current['assigned_class_name']}")

    results["objects"] = all_objects

    with gzip.open(input_file, "wb") as f:
        pickle.dump(results, f)
    print(f"Total valid objects: {total_valid_objects}/{len(all_objects)}")
    print(f"Refined names saved to file: {input_file}")


main()
