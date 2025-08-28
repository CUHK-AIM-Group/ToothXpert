import argparse
import os
import re
import sys

import bleach
import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from model.ToothXpert import ToothXpertForCausalLM
from model.ToothXpert_MOE import ToothXpertForCausalLMMOE
from mypeft import LoraConfig, get_peft_model
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)


def parse_args(args):
    parser = argparse.ArgumentParser(description="ToothXpert chat")
    parser.add_argument("--version", default="/path/to/saved/hf/model", type=str)
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    parser.add_argument("--moe_lora", default=True, action="store_true")
    parser.add_argument("--expert_num", default=3, type=int)
    parser.add_argument("--guide", default=True, action="store_true")
    parser.add_argument("--guide_mode", default="smmulsm", type=str)
    return parser.parse_args(args)


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


args = parse_args(sys.argv[1:])
os.makedirs(args.vis_save_path, exist_ok=True)

# Create model
tokenizer = AutoTokenizer.from_pretrained(
    args.version,
    cache_dir=None,
    model_max_length=args.model_max_length,
    padding_side="right",
    use_fast=False,
)
tokenizer.pad_token = tokenizer.unk_token
num_added_tokens = tokenizer.add_tokens("[SEG]")
args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

if args.use_mm_start_end:
    tokenizer.add_tokens(
        [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
    )

moe_lora_args = {
    "lora_r": args.lora_r,
    "lora_alpha": args.lora_alpha,
    "lora_dropout": args.lora_dropout,
    "lora_target_modules": args.lora_target_modules,
    "moe_lora": args.moe_lora,
    "expert_num": args.expert_num,
    "guide": args.guide,
    "guide_mode": args.guide_mode,
    "vocab_size": len(tokenizer),
}


torch_dtype = torch.float32
if args.precision == "bf16":
    torch_dtype = torch.bfloat16
elif args.precision == "fp16":
    torch_dtype = torch.half

kwargs = {"torch_dtype": torch_dtype,
    "train_mask_decoder": args.train_mask_decoder,
    "out_dim": args.out_dim,
    "moe_lora_args": moe_lora_args,
}

model = ToothXpertForCausalLMMOE.from_pretrained(
    args.version, low_cpu_mem_usage=True, vision_tower=args.vision_tower, seg_token_idx=args.seg_token_idx, **kwargs
)

model.config.eos_token_id = tokenizer.eos_token_id
model.config.bos_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

model.get_model().initialize_vision_modules(model.get_model().config)
vision_tower = model.get_model().get_vision_tower()
vision_tower.to(dtype=torch_dtype)
# model.get_model().initialize_toothxpert_modules(model.get_model().config)

if args.precision == "bf16":
    model = model.bfloat16().cuda()
elif (
    args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
):
    vision_tower = model.get_model().get_vision_tower()
    model.model.vision_tower = None
    import deepspeed

    model_engine = deepspeed.init_inference(
        model=model,
        dtype=torch.half,
        replace_with_kernel_inject=True,
        replace_method="auto",
    )
    model = model_engine.module
    model.model.vision_tower = vision_tower.half().cuda()
elif args.precision == "fp32":
    model = model.float().cuda()

vision_tower = model.get_model().get_vision_tower()
vision_tower.to(device=args.local_rank)

clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
transform = ResizeLongestSide(args.image_size)

model.eval()

# Gradio
examples = [
    [
        "Can you describe the image for me?",
        "/data/xyliu/images_resized/T_P2_62.png",
    ],
    [
        "What do you see in the panoramic radiograph?",
        "/data/xyliu/images_resized/T_P4_334.jpg",
    ],
    [
        "Can you segment the teeth in the image?",
        "/data/xyliu/images_resized/T_P2_189.png",
    ],
    # [
    #     "Can you describe the image for me?",
    #     "./vis_output/T_P4_326.jpg",
    # ],
    # [
    #     "What can you see in the dental radiograph?",
    #     "/home/xyliu/content/ToothXpert/vis_output/T_P2_543.png",
    # ],
]
output_labels = ["Segmentation Output"]

title = "ToothXpert: A Large Vision-Language Model for Panoramic Dental X-Ray Interpretation"

description = """
<font size=4>
This is the online demo of ToothXpert. \n
If multiple users are using it at the same time, they will enter a queue, which may delay some time. \n
**Note**: **Different prompts can lead to significantly varied results**. \n
**Note**: Please try to **standardize** your input text prompts to **avoid ambiguity**, and also pay attention to whether the **punctuations** of the input are correct. \n
**Usage**: <br>
&ensp;(1) To let ToothXpert **segment the teeth**, input prompt like: "Can you segment the teeth in this image?"; <br>
&ensp;(2) To let ToothXpert **output a diagnosis report**, input prompt like: "Can you describe the image for me?" or "What can you see in the image?"; <br>
&ensp;(3) To obtain **answer for a specific symptom**, you can input "Is there any non-metalic restorations in the image?". <br>
Hope you can enjoy our work!
</font>
"""

# article = """
# <p style='text-align: center'>
# <a href='https://github.com' target='_blank'>
# Preprint Paper
# </a>
# \n
# <p style='text-align: center'>
# <a href='https://github.com' target='_blank'>   Github Repo </a></p>
# """

article = """
"""

## to be implemented
def inference(input_str, input_image):
    ## filter out special chars
    input_str = bleach.clean(input_str)

    print("input_str: ", input_str, "input_image: ", input_image)

    # Model Inference
    conv = conversation_lib.conv_templates[args.conv_type].copy()
    conv.messages = []

    prompt = input_str
    prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    if args.use_mm_start_end:
        replace_token = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        )
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()

    image_np = cv2.imread(input_image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    original_size_list = [image_np.shape[:2]]

    image_clip = (
        clip_image_processor.preprocess(image_np, return_tensors="pt")[
            "pixel_values"
        ][0]
        .unsqueeze(0)
        .cuda()
    )
    if args.precision == "bf16":
        image_clip = image_clip.bfloat16()
    elif args.precision == "fp16":
        image_clip = image_clip.half()
    else:
        image_clip = image_clip.float()

    image = transform.apply_image(image_np)
    resize_list = [image.shape[:2]]

    image = (
        preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        .unsqueeze(0)
        .cuda()
    )
    if args.precision == "bf16":
        image = image.bfloat16()
    elif args.precision == "fp16":
        image = image.half()
    else:
        image = image.float()


    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda()

    output_ids, pred_masks = model.evaluate(
        image_clip,
        image,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=2048,
        tokenizer=tokenizer,
    )
    output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

    text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
    # text_output = text_output.replace("\n", "").replace("  ", " ")
    text_output = text_output.replace("  ", " ").replace("</s>", "")
    text_output = text_output.split("ASSISTANT: ")[-1]
    # in fact we have to handle some mis spelling or class name incoherence,
    # but this is optional, if you only want to get a single result instead of evaluation the metrics.
    text_output = text_output.replace("amalgam restorations", "metallic restorations")
    text_output = text_output.replace("teeth supported bridge", "teeth supported bridge/crown")
    text_output = text_output.replace("teeth supported crown", "teeth supported bridge/crown")
    text_output = text_output.replace("implant supported bridge", "implant supported bridge/crown")
    text_output = text_output.replace("implant supported crown", "implant supported bridge/crown")
    # if occur twice
    # count implant supported bridge/crown
    if text_output.count("implant supported bridge/crown") > 1:
        # remove the second one
        text_output = text_output.replace("* implant supported bridge/crown;", "", 1)
    if text_output.count("teeth supported bridge/crown") > 1:
        # remove the second one
        text_output = text_output.replace("* teeth supported bridge/crown;", "", 1)

    print("text_output: ", text_output)
    save_img = None
    for i, pred_mask in enumerate(pred_masks):
        if pred_mask.shape[0] == 0:
            continue

        pred_mask = pred_mask.detach().cpu().numpy()[0]
        pred_mask = pred_mask > 0

        save_img = image_np.copy()
        save_img[pred_mask] = (
            image_np * 0.5
            + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
        )[pred_mask]
    # this is to make the correct correspondence that R/L is radiolucency
    text_output = text_output.replace("right or left", "radiolucency")
    if text_output[-1] == ";":
        text_output = text_output[:-1] + "."
    output_str = "ASSISTANT: " + text_output  # input_str
    if save_img is not None:
        output_image = save_img  # input_image
    else:
        ## no seg output
        output_image = cv2.imread('./vis_output/white.png')[:, :, ::-1]
    return output_image, output_str


demo = gr.Interface(
    inference,
    [
        gr.Textbox(lines=1, placeholder=None, label="Text Instruction"),
        gr.Image(type="filepath", label="Input Image"),
    ],
    # outputs=[
    #     gr.Image(type="pil", label="Segmentation Output"),
    #     gr.Textbox(lines=5, placeholder=None, label="Text Output"),
    # ],
    [
        "image",
        "text"
    ],
    title=title,
    description=description,
    article=article,
    examples=examples,
    allow_flagging="auto",
)

demo.queue()
demo.launch(share=True)
