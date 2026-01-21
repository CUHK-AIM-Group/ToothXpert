#!/usr/bin/env python
"""
ToothXpert Single Image Inference
Tests 1 summary + 11 condition questions on a dental X-ray image

Usage:
    # Use default image (T_P6_59.png)
    python test_comprehensive.py

    # Test specific image
    python test_comprehensive.py --image_path /path/to/your/image.png

    # Specify custom model path
    python test_comprehensive.py --model_path /path/to/model --image_path /path/to/image.png

    # Use different GPU
    python test_comprehensive.py --device cuda:1

Questions tested:
    1. Summary: Describes the overall dental X-ray findings
    2-12. Conditions: Binary yes/no questions about specific dental conditions
        - Amalgam restorations
        - Caries (R/L)
        - Crestal bone loss (mandible/maxillary)
        - Implant-supported bridge
        - Dental implant
        - Metallic/non-metallic post
        - Non-metallic restorations
        - Periapical radiolucency
        - Root canal treated teeth
        - Tooth-supported bridge
"""
import argparse
import os
import sys
import warnings
import cv2
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPImageProcessor

# Suppress warnings and verbose output
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from model.ToothXpert_MOE import ToothXpertForCausalLMMOE
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def run_inference(model, tokenizer, image_np, question, device='cuda:0'):
    """Run inference for a single question"""
    import io
    from contextlib import redirect_stdout, redirect_stderr

    original_size_list = [image_np.shape[:2]]

    # Prepare CLIP input (suppress verbose loading)
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        clip_image_processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14",
            local_files_only=True,
        )
    transform = ResizeLongestSide(1024)

    image_clip = (
        clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0]
        .unsqueeze(0)
        .to(device)
        .bfloat16()
    )

    image = transform.apply_image(image_np)
    resize_list = [image.shape[:2]]
    image = (
        preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        .unsqueeze(0)
        .to(device)
        .bfloat16()
    )

    # Prepare prompt
    conv = conversation_lib.conv_templates["llava_v1"].copy()
    conv.messages = []
    prompt = DEFAULT_IMAGE_TOKEN + "\n" + question
    prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)

    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        output_ids, _ = model.evaluate(  # pred_masks not used
            image_clip,
            image,
            input_ids,
            resize_list,
            original_size_list,
            max_new_tokens=512,
            tokenizer=tokenizer,
        )

    output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
    text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
    text_output = text_output.replace("\n", "").replace("  ", " ")
    text_output = text_output.split('ASSISTANT:')[-1].replace('</s>', '').strip()

    return text_output


def parse_args():
    parser = argparse.ArgumentParser(description="ToothXpert Single Image Inference")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./ToothXpert",
        help="Path to the ToothXpert model"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="./demo/example_image_1.png",
        help="Path to the dental X-ray image to analyze"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run inference on (default: cuda:0)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("ToothXpert Single Image Inference")
    print("Summary + 11 Dental Condition Questions")
    print("=" * 80)

    # Hardcoded 12 questions (1 summary + 11 conditions)
    test_questions = [
        "Can you describe the image for me?",  # Summary
        "Is there any amalgam restorations in the image?",
        "Any R/L suggestive of caries present?",
        "Is there any generalised crestal bone loss of the mandible in the image?",
        "Is there any generalised crestal bone loss of the maxillary in the image?",
        "Is there any implant-supported bridge present?",
        "Is there any dental implant present?",
        "Is there any metallic and non-metallic post present?",
        "Is there any non-metalic restorations in the image?",
        "Is there any periapical R/L observed on tooth?",
        "Is there any root canal treated teeth?",
        "Is there any tooth supported bridge present?",
    ]

    # Check image path
    image_path = args.image_path
    if not os.path.exists(image_path):
        print(f"✗ ERROR: Image not found: {image_path}")
        sys.exit(1)

    print(f"\n✓ Image: {image_path}")
    print(f"  Testing {len(test_questions)} questions (1 summary + 11 conditions)")

    # Check CUDA availability
    if not torch.cuda.is_available() and args.device.startswith('cuda'):
        print("\n✗ ERROR: CUDA not available but cuda device specified!")
        sys.exit(1)

    # Load tokenizer
    print("\nLoading tokenizer...")
    import io
    from contextlib import redirect_stdout, redirect_stderr

    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            model_max_length=512,
            padding_side="right",
            use_fast=False,
            local_files_only=True,
        )
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.add_tokens("[SEG]")
        seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    print("✓ Tokenizer loaded")

    # Load model
    print("\nLoading model (this takes 2-3 minutes)...")
    sys.stdout.flush()

    moe_lora_args = {
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_target_modules": "q_proj,v_proj",
        "moe_lora": False,
        "expert_num": 3,
        "guide": True,
        "guide_mode": "smmulsm",
        "vocab_size": len(tokenizer),
    }

    kwargs = {
        "torch_dtype": torch.bfloat16,
        "train_mask_decoder": True,
        "out_dim": 256,
        "moe_lora_args": moe_lora_args,
    }

    # Suppress verbose output during model loading
    import io
    from contextlib import redirect_stdout, redirect_stderr

    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        model = ToothXpertForCausalLMMOE.from_pretrained(
            args.model_path,
            low_cpu_mem_usage=True,
            vision_tower="openai/clip-vit-large-patch14",
            seg_token_idx=seg_token_idx,
            local_files_only=False,
            **kwargs
        )

        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

        model.get_model().initialize_vision_modules(model.get_model().config)
        vision_tower = model.get_model().get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16)

        model = model.bfloat16().to(args.device)
        vision_tower.to(device=args.device)
        model.eval()

    print(f"✓ Model loaded and ready on {args.device}")

    # Load image
    print("\nLoading image...")
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    print(f"✓ Image loaded: {image_np.shape}")

    # Run inference for all questions
    n_qa = len(test_questions)
    print(f"\n{'=' * 80}")
    print(f"Running {n_qa} inferences (1 summary + 11 conditions)...")
    print(f"{'=' * 80}\n")

    results = []
    for qa_idx, question in enumerate(test_questions):
        # Detect if this is a summary question
        is_summary = "describe" in question.lower() or "summary" in question.lower()

        print(f"[{qa_idx + 1}/{n_qa}] {'SUMMARY' if is_summary else f'Condition {qa_idx}'}")
        print(f"Q: {question[:80]}..." if len(question) > 80 else f"Q: {question}")
        sys.stdout.flush()

        # Run inference
        prediction = run_inference(model, tokenizer, image_np, question, device=args.device)

        results.append({
            'idx': qa_idx,
            'type': 'Summary' if is_summary else f'Condition {qa_idx}',
            'question': question,
            'prediction': prediction,
        })

        print(f"A: {prediction}")
        print()

    # Summary
    print("=" * 80)
    print("SUMMARY OF RESULTS")
    print("=" * 80)

    # Separate summary and conditions
    summary_results = [r for r in results if r['type'] == 'Summary']
    condition_results = [r for r in results if r['type'] != 'Summary']

    if summary_results:
        print(f"\n[Summary Question]")
        r = summary_results[0]
        print(f"Q: {r['question']}")
        print(f"\nPrediction:")
        print(f"{r['prediction'][:400]}..." if len(r['prediction']) > 400 else f"{r['prediction']}")
        print()

    if condition_results:
        print(f"\n[Condition Questions: {len(condition_results)} total]")
        print()
        for r in condition_results:
            print(f"{r['idx']}. {r['question']}")
            print(f"   Answer: {r['prediction']}")
            print()

    print("=" * 80)
    print("✓ Comprehensive test completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
