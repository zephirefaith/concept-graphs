'''
Acknowledgements: 
This code is adapted from the LLaVA repository by Krishna Murthy Jatavallabhula 
'''

import argparse
import os
import pickle as pkl
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPVisionModel,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModel,
)
from transformers import logging as hf_logging
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

try:
    LLAVA_PYTHON_PATH = os.environ["LLAVA_PYTHON_PATH"]
except KeyError:
    print("Please set the environment variable LLAVA_PYTHON_PATH to the path of the LLaVA repository")
    sys.exit(1)
    
sys.path.append(LLAVA_PYTHON_PATH)

from llava.conversation import SeparatorStyle, conv_templates
# from llava.model.utils import KeywordsStoppingCriteria
from llava.mm_utils import KeywordsStoppingCriteria
from llava.utils import disable_torch_init

torch.autograd.set_grad_enabled(False)

# Set logging verbosity for the transformers package to only log errors
hf_logging.set_verbosity_error()


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModelTweaked(LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super().__init__(config)

        # Instantiate the multimodal vision tower
        if hasattr(config, "mm_vision_tower"):
            # HACK: for FSDP (fully-sharded data parallel); cast to a list
            self.vision_tower = [
                CLIPVisionModel.from_pretrained(config.mm_vision_tower)
            ]

        # Instantiate the multimodal linear projection (vision-language)
        if hasattr(config, "use_mm_proj"):
            self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)

    def get_vision_tower(self):
        # Return the vision tower
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(
        self,
        vision_tower,
        mm_vision_select_layer,
        pretrain_mm_mlp_adapter=None,
        fsdp=None,
    ):
        self.config.mm_vision_tower = vision_tower

        image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        if not hasattr(self, "vision_tower"):
            vision_tower = CLIPVisionModel.from_pretrained(vision_tower)
        else:
            vision_tower = self.vision_tower[0]
        # Vision tower is frozen
        vision_tower.requires_grad_(False)

        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower = [vision_tower]
        else:
            self.vision_tower = vision_tower

        vision_config = vision_tower.config
        num_patches = (vision_config.image_size // vision_config.patch_size) ** 2

        # If using a vision tower, we need a multimodal projection (from vision to text)
        self.config.use_mm_proj = True
        self.config.mm_hidden_size = vision_config.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer

        if not hasattr(self, "mm_projector"):
            self.mm_projector = nn.Linear(
                vision_config.hidden_size, self.config.hidden_size
            )

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(
                pretrain_mm_mlp_adapter, map_location="cpu"
            )
            self.mm_projector.load_state_dict(
                {k.split(".")[-1]: v for k, v in mm_projector_weights.items()}
            )

        return dict(
            image_processor=image_processor,
            image_token_len=num_patches,
            vision_config=vision_config,
        )

    def encode_image(
        self,
        images: Union[List, torch.FloatTensor],
    ) -> Union[list, torch.FloatTensor]:
        with torch.no_grad():
            vision_tower = self.get_vision_tower()
            if vision_tower is not None and images is not None:
                if type(images) is list:
                    image_features = []
                    for image in images:
                        image_forward_out = vision_tower(
                            image.unsqueeze(0), output_hidden_states=True
                        )
                        select_hidden_state_layer = getattr(
                            self.config, "mm_vision_select_layer", -1
                        )
                        # TODO (Krishna): Figure out why the zero-th dim is dropped (perhaps that is the
                        # global token?)
                        image_feature = select_hidden_state[:, 1:]
                        image_features.append(image_feature)
                    # image_features = [self.mm_projector(image_feature)[0] for image_feature in image_features]
                else:
                    image_forward_outs = vision_tower(
                        images.to(vision_tower.dtype), output_hidden_states=True
                    )
                    select_hidden_state_layer = getattr(
                        self.config, "mm_vision_select_layer", -1
                    )
                    select_hidden_state = image_forward_outs.hidden_states[
                        select_hidden_state_layer
                    ]
                    image_features = select_hidden_state[:, 1:].to(images.dtype)
                    # image_features = self.mm_projector(image_features)
            return image_features

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[Union[list, torch.FloatTensor]] = None,
        image_features: Optional[Union[list, torch.FloatTensor]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # HACK: replace back original embeddings for LLaVA pretraining
        orig_embeds_params = getattr(self, "orig_embeds_params", None)
        # if orig_embeds_params is not None:
        #     orig_embeds_params = orig_embeds_params[0]
        #     with torch.no_grad():
        #         self.get_input_embeddings().weight.data[:-2] = orig_embeds_params[:-2].data

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        vision_tower = self.get_vision_tower()
        if vision_tower is not None and (input_ids.shape[1] != 1 or self.training):
            # If images are provided, but image features is not provided, compute the image_features
            # from the images passed in. (else, skip this step)
            if image_features is None and images is not None:
                # TODO: this is a modified multimodal LLM -- Haotian Liu
                with torch.no_grad():
                    if type(images) is list:
                        # variable length images
                        image_features = []
                        for image in images:
                            image_forward_out = vision_tower(
                                image.unsqueeze(0), output_hidden_states=True
                            )
                            select_hidden_state_layer = getattr(
                                self.config, "mm_vision_select_layer", -1
                            )
                            select_hidden_state = image_forward_out.hidden_states[
                                select_hidden_state_layer
                            ]
                            image_feature = select_hidden_state[:, 1:]
                            image_features.append(image_feature)
                    else:
                        image_forward_outs = vision_tower(
                            images.to(vision_tower.dtype), output_hidden_states=True
                        )
                        select_hidden_state_layer = getattr(
                            self.config, "mm_vision_select_layer", -1
                        )
                        select_hidden_state = image_forward_outs.hidden_states[
                            select_hidden_state_layer
                        ]
                        image_features = select_hidden_state[:, 1:].to(images.dtype)
                if type(images) is list:
                    image_features = [
                        self.mm_projector(image_feature)[0]
                        for image_feature in image_features
                    ]
                else:
                    image_features = self.mm_projector(image_features)

            if type(image_features) is list:
                image_features = [
                    self.mm_projector(image_feature)[0]
                    for image_feature in image_features
                ]
            elif image_features is not None:  # image_features is torch.tensor
                image_features = self.mm_projector(image_features)

            # TODO (Krishna): The following line from the original LLaVA impl seems like a hardcoded param
            # size. Replace this by reading in these dims from the config file in future.
            dummy_image_features = torch.zeros(
                256, 1024, device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )
            dummy_image_features = self.mm_projector(dummy_image_features)
            # (Krishna): If no image_features are passed as input, feed the dummy image features through
            if image_features is None:
                image_features = dummy_image_features

            new_input_embeds = []
            cur_image_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                if (cur_input_ids == vision_tower.config.im_patch_token).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = (
                        cur_input_embeds + (0.0 * dummy_image_features).sum()
                    )
                    new_input_embeds.append(cur_input_embeds)
                    cur_image_idx += 1
                    continue
                if vision_tower.config.use_im_start_end:
                    cur_image_features = image_features[cur_image_idx]
                    num_patches = cur_image_features.shape[0]
                    if (cur_input_ids == vision_tower.config.im_start_token).sum() != (
                        cur_input_ids == vision_tower.config.im_end_token
                    ).sum():
                        raise ValueError(
                            "The number of image start tokens and image end tokens should be the same."
                        )
                    image_start_tokens = torch.where(
                        cur_input_ids == vision_tower.config.im_start_token
                    )[0]
                    for image_start_token_pos in image_start_tokens:
                        cur_image_features = image_features[cur_image_idx].to(
                            device=cur_input_embeds.device
                        )
                        num_patches = cur_image_features.shape[0]
                        if (
                            cur_input_ids[image_start_token_pos + num_patches + 1]
                            != vision_tower.config.im_end_token
                        ):
                            raise ValueError(
                                "The image end token should follow the image start token."
                            )
                        if orig_embeds_params is not None:
                            cur_new_input_embeds = torch.cat(
                                (
                                    cur_input_embeds[:image_start_token_pos].detach(),
                                    cur_input_embeds[
                                        image_start_token_pos : image_start_token_pos
                                        + 1
                                    ],
                                    cur_image_features,
                                    cur_input_embeds[
                                        image_start_token_pos
                                        + num_patches
                                        + 1 : image_start_token_pos
                                        + num_patches
                                        + 2
                                    ],
                                    cur_input_embeds[
                                        image_start_token_pos + num_patches + 2 :
                                    ].detach(),
                                ),
                                dim=0,
                            )
                        else:
                            cur_new_input_embeds = torch.cat(
                                (
                                    cur_input_embeds[: image_start_token_pos + 1],
                                    cur_image_features,
                                    cur_input_embeds[
                                        image_start_token_pos + num_patches + 1 :
                                    ],
                                ),
                                dim=0,
                            )
                        cur_image_idx += 1
                    new_input_embeds.append(cur_new_input_embeds)
                else:
                    cur_image_features = image_features[cur_image_idx]
                    num_patches = cur_image_features.shape[0]
                    if (
                        cur_input_ids == vision_tower.config.im_patch_token
                    ).sum() != num_patches:
                        raise ValueError(
                            "The number of image patch tokens should be the same as the number of image patches."
                        )
                    masked_indices = torch.where(
                        cur_input_ids == vision_tower.config.im_patch_token
                    )[0]
                    mask_index_start = masked_indices[0]
                    if (
                        masked_indices
                        != torch.arange(
                            mask_index_start,
                            mask_index_start + num_patches,
                            device=masked_indices.device,
                            dtype=masked_indices.dtype,
                        )
                    ).any():
                        raise ValueError(
                            "The image patch tokens should be consecutive."
                        )
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat(
                            (
                                cur_input_embeds[:mask_index_start].detach(),
                                cur_image_features,
                                cur_input_embeds[
                                    mask_index_start + num_patches :
                                ].detach(),
                            ),
                            dim=0,
                        )
                    else:
                        cur_new_input_embeds = torch.cat(
                            (
                                cur_input_embeds[:mask_index_start],
                                cur_image_features,
                                cur_input_embeds[mask_index_start + num_patches :],
                            ),
                            dim=0,
                        )
                    new_input_embeds.append(cur_new_input_embeds)
                    cur_image_idx += 1
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class LlavaLlamaForCausalLMTweaked(LlamaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = LlavaLlamaModelTweaked(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_vision_tower(self):
        model = self.get_model()
        vision_tower = model.vision_tower
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[Union[list, torch.FloatTensor]] = None,
        image_features: Optional[Union[list, torch.FloatTensor]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            images=images,
            image_features=image_features,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "image_features": kwargs.get("image_features", None),
            }
        )
        return model_inputs

    def initialize_vision_tokenizer(
        self,
        mm_use_im_start_end,
        tokenizer,
        device,
        tune_mm_mlp_adapter=False,
        pretrain_mm_mlp_adapter=None,
    ):
        vision_config = self.get_vision_tower().config
        vision_config.use_im_start_end = mm_use_im_start_end
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        if mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
            self.resize_token_embeddings(len(tokenizer))
            (
                vision_config.im_start_token,
                vision_config.im_end_token,
            ) = tokenizer.convert_tokens_to_ids(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]
            )

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if tune_mm_mlp_adapter:
                self.get_model().orig_embeds_params = [
                    self.get_input_embeddings().weight.data.clone().to(device=device)
                ]
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(
                    pretrain_mm_mlp_adapter, map_location="cpu"
                )
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[
                        -num_new_tokens:
                    ]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. "
                        f"Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )

        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_IMAGE_PATCH_TOKEN]
        )[0]

from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.model.builder import load_pretrained_model
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle

class LLaVaChat(object):

    def __init__(self, model_path, model_base, conv_mode=None):
        # Model
        disable_torch_init()

        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, self.model_name
        )
        
        self.conv_mode = None
        if conv_mode is not None:
            self.conv_mode = conv_mode
        else:
            if 'llama-2' in self.model_name.lower():
                self.conv_mode = "llava_llama_2"
            elif "v1" in self.model_name.lower():
                self.conv_mode = "llava_v1"
            elif "mpt" in self.model_name.lower():
                self.conv_mode = "mpt"
            else:
                self.conv_mode = "llava_v0"

        self.reset()

    def reset(self):
        # Initialize a conversation from template (default conv_mode is "multimodal")
        # (conv_mode determines the conversation template to use from llava.conversation module)
        self.conv = conv_templates[self.conv_mode].copy()

        # Cache for image features
        self.image_features = None
    
    def __call__(self, query, image_features=None):

        qs = query
        if image_features is not None:
            if self.model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        else:
            qs = qs + '\n'
        
        if self.image_features is None:
            self.image_features = image_features
        
        self.conv.append_message(self.conv.roles[0], qs)
        self.conv.append_message(self.conv.roles[1], None)

        input_ids = None
        # Get the prompt
        prompt = self.conv.get_prompt()
        # Tokenize this prompt
        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).cuda()

        stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=None,
                image_features=self.image_features,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        self.conv.append_message(self.conv.roles[1], outputs)
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        return outputs
    
    def load_image(self, image_file):
        if image_file.startswith('http') or image_file.startswith('https'):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_file).convert('RGB')
        return image
    
    def encode_image(self, image_tensor_half_cuda):
        return self.model.encode_images(image_tensor_half_cuda)


# class LLaVaChat(object):
#     def __init__(self, model_path, conv_mode="multimodal", num_gpus=1):
#         self.model_path = model_path
#         self.conv_mode = conv_mode
#         self.num_gpus = num_gpus
# 
#         # Handle multi-gpu config
#         if self.num_gpus == 1:
#             kwargs = {}
#         else:
#             kwargs = {
#                 "device_map": "auto",
#                 "max_memory": {i: "13GiB" for i in range(self.num_gpus)},
#             }
# 
#         # pytorch spends a substantial amount of time initializing default weights for
#         # each linear layer and layernorm layer in the model. Since we will load weights
#         # from disk anyways, disable this redundant default init.
#         # This accelerates model creation.
#         disable_torch_init()
# 
#         # Initialize the tokenizer
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
#         # Initialize the model
#         self.model = LlavaLlamaForCausalLMTweaked.from_pretrained(
#             self.model_path, torch_dtype=torch.float16, low_cpu_mem_usage=False, **kwargs
#         )
#         self.model.cuda()
# 
#         # Image preprocessor
#         self.image_processor = CLIPImageProcessor.from_pretrained(
#             self.model.config.mm_vision_tower, torch_dtype=torch.float16
#         )
# 
#         self.mm_use_im_start_end = getattr(
#             self.model.config, "mm_use_im_start_end", False
#         )
#         self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
#         if self.mm_use_im_start_end:
#             self.tokenizer.add_tokens(
#                 [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
#             )
# 
#         self.vision_tower = self.model.get_model().vision_tower[0]
#         if self.vision_tower.device.type == "meta":
#             self.vision_tower = CLIPVisionModel.from_pretrained(
#                 self.vision_tower.config._name_or_path,
#                 torch_dtype=torch.float16,
#                 low_cpu_mem_usage=True,
#             ).cuda()
#             self.model.get_model().vision_tower[0] = self.vision_tower
#         else:
#             self.vision_tower.to(device="cuda", dtype=torch.float16)
#         self.vision_config = self.vision_tower.config
#         self.vision_config.im_patch_token = self.tokenizer.convert_tokens_to_ids(
#             [DEFAULT_IMAGE_PATCH_TOKEN]
#         )[0]
#         self.vision_config.use_im_start_end = self.mm_use_im_start_end
#         if self.mm_use_im_start_end:
#             (
#                 self.vision_config.im_start_token,
#                 self.vision_config.im_end_token,
#             ) = self.tokenizer.convert_tokens_to_ids(
#                 [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]
#             )
#         self.image_token_len = (
#             self.vision_config.image_size // self.vision_config.patch_size
#         ) ** 2
# 
#         # # Initialize a conversation from template (default conv_mode is "multimodal")
#         # # (conv_mode determines the conversation template to use from llava.conversation module)
#         # self.conv = conv_templates[self.conv_mode].copy()
# 
#         # # Cache for image features
#         # self.image_features = None
#         
#         self.reset()
#         
#     def reset(self):
#         # Initialize a conversation from template (default conv_mode is "multimodal")
#         # (conv_mode determines the conversation template to use from llava.conversation module)
#         self.conv = conv_templates[self.conv_mode].copy()
#         
#         # Cache for image features
#         self.image_features = None
# 
#     def __call__(self, query, image_features=None):
#         # Given this query, and the image_featurese, prompt LLaVA with the query,
#         # using the image_features as context.
# 
#         qs = query
#         if image_features is not None:
#             if self.mm_use_im_start_end:
#                 qs = (
#                     qs
#                     + "\n"
#                     + DEFAULT_IM_START_TOKEN
#                     + DEFAULT_IMAGE_PATCH_TOKEN * self.image_token_len
#                     + DEFAULT_IM_END_TOKEN
#                 )
#             else:
#                 qs = qs + "\n" + DEFAULT_IMAGE_PATCH_TOKEN * self.image_token_len
#             if self.image_features is None:
#                 self.image_features = image_features
#         else:
#             qs = qs + "\n"
# 
#         self.conv.append_message(self.conv.roles[0], qs)
#         self.conv.append_message(self.conv.roles[1], None)
# 
#         input_ids = None
# 
#         # Get the prompt
#         prompt = self.conv.get_prompt()
#         # Tokenize this prompt
#         inputs = self.tokenizer([prompt])
#         # Cast to torch tensor and to GPU
#         input_ids = torch.as_tensor(inputs.input_ids).cuda()
# 
#         stop_str = (
#             self.conv.sep
#             if self.conv.sep_style != SeparatorStyle.TWO
#             else self.conv.sep2
#         )
#         keywords = [stop_str]
#         stopping_criteria = KeywordsStoppingCriteria(
#             keywords, self.tokenizer, input_ids
#         )
# 
#         with torch.inference_mode():
#             output_ids = self.model.generate(
#                 input_ids,
#                 image_features=self.image_features,
#                 do_sample=True,
#                 temperature=0.2,
#                 max_new_tokens=1024,
#                 stopping_criteria=[stopping_criteria],
#             )
# 
#         input_token_len = input_ids.shape[1]
#         n_diff_input_output = (
#             (input_ids != output_ids[:, :input_token_len]).sum().item()
#         )
#         if n_diff_input_output > 0:
#             print(
#                 f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
#             )
#         outputs = self.tokenizer.batch_decode(
#             output_ids[:, input_token_len:], skip_special_tokens=True
#         )[0]
#         self.conv.append_message(self.conv.roles[1], outputs)
#         outputs = outputs.strip()
#         if outputs.endswith(stop_str):
#             outputs = outputs[: -len(stop_str)]
#         outputs = outputs.strip()
#         return outputs
# 
#     def load_image(self, image_file):
#         if image_file.startswith("http") or image_file.startswith("https"):
#             response = requests.get(image_file)
#             image = Image.open(BytesIO(response.content)).convert("RGB")
#         else:
#             image = Image.open(image_file).convert("RGB")
#         return image
# 
#     def encode_image(self, image_tensor_half_cuda):
#         return self.model.model.encode_image(image_tensor_half_cuda)
    



if __name__ == "__main__":
    # from types import SimpleNamespace

    # args = SimpleNamespace()
    # args.model_path = "/home/krishna/code/LLaVA/LLaVA-7b-v0"
    # args.image_file = "/home/krishna/Downloads/figurines/images/frame_00150.jpg"
    # args.conv_mode = "multimodal"
    # args.num_gpus = 1
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--image_file", type=str, required=True)
    parser.add_argument("--conv_mode", type=str, default="multimodal")
    parser.add_argument("--num_gpus", type=int, default=1)
    args = parser.parse_args()
    
    if args.model_path is None:
        try: 
            args.model_path = os.environ["LLAVA_CKPT_PATH"]
        except KeyError:
            print("Please provide a model path or set the environment variable LLAVA_CKPT_PATH")
            exit(1)

    chat = LLaVaChat(args.model_path, None)
    print("LLaVA chat initialized...")

    image = chat.load_image(args.image_file)
    image_tensor = chat.image_processor.preprocess(image, return_tensors="pt")[
        "pixel_values"
    ][0]
    image_features = chat.encode_image(image_tensor[None, ...].half().cuda())
    # query = input("Enter a query ('q' to quit): ")
    query = "List the set of objects in this image."
    outputs = chat(query=query, image_features=image_features)
    print(outputs)

    query = "List potential uses for each."
    outputs = chat(query=query, image_features=None)
    print(outputs)

    # while True:
    #     query = input("Enter a query ('q' to quit): ")
    #     # query = "Describe this image"
    #     # print(query)
    #     outputs = chat(query=query, image_features=None)
    #     print(outputs)
