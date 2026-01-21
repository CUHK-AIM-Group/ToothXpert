#!/usr/bin/env python
"""
ToothXpert Segmentation Test Script
Tests tooth segmentation capabilities using SAM integration

Usage:
    # Use default image
    python test_segmentation.py

    # Test specific image
    python test_segmentation.py --image_path /path/to/your/image.png

    # Specify custom model path
    python test_segmentation.py --model_path /path/to/model --image_path /path/to/image.png

    # Use different GPU
    python test_segmentation.py --device cuda:1

    # Custom output directory
    python test_segmentation.py --output_dir ./my_segmentation_results

Description:
    This script tests the segmentation capabilities of ToothXpert. It asks the model
    to segment teeth in a dental X-ray image and visualizes the results by overlaying
    the segmentation masks on the original image.

    The model uses SAM (Segment Anything Model) integration to produce segmentation
    masks. When prompted with segmentation questions, the model generates [SEG] tokens
    which trigger mask generation.
"""
import argparse
import os
import sys
import warnings
import cv2
import numpy as np
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


def visualize_masks(image_np, pred_masks, alpha=0.5):
    """
    Visualize segmentation masks on the original image

    Args:
        image_np: Original image (H, W, 3) in RGB
        pred_masks: List of predicted masks from the model
        alpha: Blending factor (0.5 = 50% original, 50% mask overlay)

    Returns:
        Visualized image with masks overlaid in red
    """
    if len(pred_masks) == 0:
        return image_np, 0

    save_img = image_np.copy()
    total_masks = 0

    for i, pred_mask in enumerate(pred_masks):
        if pred_mask.shape[0] == 0:
            continue

        # Convert mask to binary numpy array
        pred_mask = pred_mask.detach().cpu().numpy()[0]
        pred_mask = pred_mask > 0

        # Apply red overlay where mask is True
        save_img[pred_mask] = (
            image_np * alpha
            + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * (1 - alpha)
        )[pred_mask]

        total_masks += 1

    return save_img, total_masks


def run_segmentation_inference(model, tokenizer, image_np, question, device='cuda:0'):
    """Run inference for a segmentation question"""
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
        output_ids, pred_masks = model.evaluate(
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

    return text_output, pred_masks


def parse_args():
    parser = argparse.ArgumentParser(description="ToothXpert Segmentation Test")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./ToothXpert",
        help="Path to the ToothXpert model"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="./demo/example_image_2.png",
        help="Path to the dental X-ray image to analyze"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run inference on (default: cuda:0)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./segmentation_output",
        help="Directory to save segmentation results (default: ./segmentation_output)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("ToothXpert Segmentation Test")
    print("SAM-based Tooth Segmentation")
    print("=" * 80)

    # Segmentation prompts to test
    test_questions = [
        "Can you segment all the teeth in this image?",
        "Please segment the teeth.",
        "Show me the tooth segmentation.",
    ]

    # Check image path
    image_path = args.image_path
    if not os.path.exists(image_path):
        print(f"\n✗ ERROR: Image not found: {image_path}")
        sys.exit(1)

    print(f"\n✓ Image: {image_path}")
    print(f"  Testing {len(test_questions)} segmentation prompts")

    # Check CUDA availability
    if not torch.cuda.is_available() and args.device.startswith('cuda'):
        print("\n✗ ERROR: CUDA not available but cuda device specified!")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"✓ Output directory: {args.output_dir}")

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

    # Run segmentation inference
    print(f"\n{'=' * 80}")
    print(f"Running segmentation tests...")
    print(f"{'=' * 80}\n")

    best_result = None
    best_mask_count = 0

    for qa_idx, question in enumerate(test_questions):
        print(f"[{qa_idx + 1}/{len(test_questions)}] Segmentation Test")
        print(f"Q: {question}")
        sys.stdout.flush()

        # Run inference
        prediction, pred_masks = run_segmentation_inference(
            model, tokenizer, image_np, question, device=args.device
        )

        print(f"A: {prediction}")

        # Visualize masks
        vis_img, mask_count = visualize_masks(image_np, pred_masks)

        print(f"   Masks generated: {mask_count}")

        if mask_count > best_mask_count:
            best_mask_count = mask_count
            best_result = (question, prediction, vis_img, mask_count)

        # Save visualization
        if mask_count > 0:
            output_filename = f"seg_test_{qa_idx + 1}.png"
            output_path = os.path.join(args.output_dir, output_filename)
            cv2.imwrite(output_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
            print(f"   Saved: {output_path}")
        else:
            print(f"   No masks generated (model may not have produced [SEG] tokens)")

        print()

    # Summary
    print("=" * 80)
    print("SEGMENTATION TEST SUMMARY")
    print("=" * 80)

    if best_result is not None:
        question, prediction, vis_img, mask_count = best_result
        print(f"\nBest result: {mask_count} masks generated")
        print(f"Question: {question}")
        print(f"Response: {prediction}")

        # Save best result
        best_output_path = os.path.join(args.output_dir, "best_segmentation.png")
        cv2.imwrite(best_output_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        print(f"\n✓ Best segmentation saved: {best_output_path}")

        # Also save original for comparison
        original_output_path = os.path.join(args.output_dir, "original_image.png")
        cv2.imwrite(original_output_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        print(f"✓ Original image saved: {original_output_path}")
    else:
        print("\n⚠ No segmentation masks were generated by any prompt.")
        print("  This may indicate that:")
        print("  1. The model needs specific segmentation prompts")
        print("  2. The image may not contain segmentable objects")
        print("  3. The model may require fine-tuning for this specific task")

    print("\n" + "=" * 80)
    print("✓ Segmentation test completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
